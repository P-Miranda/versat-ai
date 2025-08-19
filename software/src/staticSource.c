#include "versat_ai.h"

#include "stdbool.h"
#include "stdint.h"

#include "iob_printf.h"

#define exit(...) ((void)0)
void clear_cache();

#define OFFSET_PTR(PTR, OFFSET) ((void *)(((char *)PTR) + OFFSET))

#define MIN(A, B) ((A < B ? A : B))
#define MAX(A, B) ((A > B ? A : B))

typedef struct {
  int addressVars[16];
  int64_t iterationDims[16];
  int64_t properDims[16]; // TODO: This is kinda ugly. We use this to handle
                          // broadcasting but we might solve the problem by
                          // either creating a new address gen that is used for
                          // broadcasting or by creating a function that takes
                          // an address gen that does the full iteration and
                          // returns an address gen that has been downsized to
                          // the value expected by the broadcasting.
  int numberDims;
} AddressGen;

void Address_Print(AddressGen *gen) {
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      printf(" x ");
    }
    printf("%d", gen->addressVars[i]);
  }

  printf(" [");
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      printf(" x ");
    }
    printf("%ld", gen->iterationDims[i]);
  }
  printf("]\n");
}

int Address_GetValue(AddressGen *gen) {
  int address = 0;
  for (int i = 0; i < gen->numberDims; i++) {
    int index = gen->addressVars[i];

    if (index >= gen->properDims[i]) {
      index = 0;
    }

    if (i > 0) {
      address *= gen->properDims[i];
    }
    address += index;
  }

  return address;
}

bool Address_IsValid(AddressGen *gen) {
  // Is this the only thing that we need?
  if (gen->addressVars[0] >= gen->iterationDims[0]) {
    return false;
  }

  for (int i = 0; i < gen->numberDims; i++) {
    if (gen->addressVars[i] >= gen->iterationDims[i]) {
      continue;
    } else {
      return true;
    }
  }
  return false;
}

void Address_Advance(AddressGen *gen) {
  if (gen->addressVars[0] >= gen->iterationDims[0]) {
    return;
  }

  for (int i = gen->numberDims - 1; i >= 0; i--) {
    if (i != 0 && gen->addressVars[i] + 1 >= gen->iterationDims[i]) {
      gen->addressVars[i] = 0;
      continue;
    } else {
      gen->addressVars[i] += 1;
      return;
    }
  }
}

AddressGen StartAddress(int64_t *iterationDims, int64_t *properDims,
                        int numberDims) {
  AddressGen gen = {};

  for (int i = 0; i < numberDims; i++) {
    gen.iterationDims[i] = iterationDims[i];
    gen.properDims[i] = properDims[i];
  }
  gen.numberDims = numberDims;

  return gen;
}

AddressGen Address_Map(AddressGen *in, int64_t *biggerDim, int *stride) {
  AddressGen gen = *in;

  for (int i = 0; i < in->numberDims; i++) {
    gen.addressVars[i] *= stride[i];
    gen.iterationDims[i] = biggerDim[i];
    gen.properDims[i] = biggerDim[i];
  }

#if 0
  printf("here\n");
  Address_Print(in);
  Address_Print(&gen);
#endif

  return gen;
}

typedef struct {
  int kernelVars[16]; // Current state

  // Kernel Info
  AddressGen *address;
  int kernelDims[16];
  int kernelDilations[16]; // NOTE: Not properly tested, do not rely on
                           // dilations being correct
  int numberDims;
} KernelGen;

KernelGen StartKernel(AddressGen *addresss, int *kernelDims, int numberDims) {
  KernelGen gen = {};
  gen.address = addresss;
  gen.kernelDims[0] = 1;
  gen.kernelDims[1] = 1;
  gen.numberDims = numberDims + 2;

  for (int i = 0; i < 16; i++) {
    gen.kernelDilations[i] = 1;
  }

  for (int i = 0; i < numberDims; i++) {
    gen.kernelDims[2 + i] = kernelDims[i];
  }

  return gen;
}

void Kernel_Print(KernelGen *gen) {
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      printf(" x ");
    }
    printf("%d", gen->kernelVars[i]);
  }

  printf(" [");
  for (int i = 0; i < gen->numberDims; i++) {
    if (i != 0) {
      printf(" x ");
    }
    printf("%ld", gen->kernelDims[i]);
  }
  printf("]\n");
}

int Kernel_GetValue(KernelGen *gen) {
  int properVars[16];

  for (int i = 0; i < gen->numberDims; i++) {
    properVars[i] = gen->kernelVars[i] * gen->kernelDilations[i] +
                    gen->address->addressVars[i];
  }

  int address = 0;
  for (int i = 0; i < gen->numberDims; i++) {
    int index = properVars[i];

    if (index >= gen->address->iterationDims[i]) {
      index = 0;
    }

    if (i > 0) {
      address *= gen->address->iterationDims[i];
    }
    address += index;
  }

  return address;
}

bool Kernel_IsValid(KernelGen *gen) {
  // Is this the only thing that we need?
  if (gen->kernelVars[0] >= gen->kernelDims[0]) {
    return false;
  }

  for (int i = 0; i < gen->numberDims; i++) {
    if (gen->kernelVars[i] >= gen->kernelDims[i]) {
      continue;
    } else {
      return true;
    }
  }
  return false;
}

bool Kernel_IsInsidePad(KernelGen *gen) {
  int properVars[16];

  for (int i = 0; i < gen->numberDims; i++) {
    properVars[i] = gen->kernelVars[i] * gen->kernelDilations[i] +
                    gen->address->addressVars[i];
  }

  for (int i = 0; i < gen->numberDims; i++) {
    if (properVars[i] >= gen->address->properDims[i]) {
      return true;
    }
  }

  return false;
}
void Kernel_Advance(KernelGen *gen) {
  if (gen->kernelVars[0] >= gen->kernelDims[0]) {
    return;
  }

  for (int i = gen->numberDims - 1; i >= 0; i--) {
    if (i != 0 && gen->kernelVars[i] + 1 >= gen->kernelDims[i]) {
      gen->kernelVars[i] = 0;
      continue;
    } else {
      gen->kernelVars[i] += 1;
      return;
    }
  }
}

void *Software_Conv(void *inputX, void *inputW, void *output, int index,
                    ConvInfo *info) {
  int batchSize = info->inputDims[0];
  int inChannels = info->inputDims[1];
  int inW = info->inputDims[3];
  int inH = info->inputDims[2];

  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  int featureMaps = info->featureMaps;

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  int outChannels = info->outputDims[1]; // Should be equal to feature maps.
  int outW = info->outputDims[3];
  int outH = info->outputDims[2];

  //printf("%d %d %d %d\n",outChannels,outH,outW,inChannels);
  //printf("%d %d\n",kernelH,kernelW);

  float *input = (float *)inputX;
  float *kernel = (float *)inputW;
  float *outView = (float *)output;

  bool print = false;

  for (int outC = 0; outC < outChannels; outC++) {
    for (int y = 0; y < outH; y++) {
      for (int x = 0; x < outW; x++) {
        float accum = 0.0f;
        for (int inC = 0; inC < inChannels; inC++) {
          if (print)
            printf("Going to accumulate: %d %d\n", y, x);

          for (int ky = 0; ky < kernelH; ky++) {
            for (int kx = 0; kx < kernelW; kx++) {
              int deltaX = x + kx;
              int deltaY = y + ky;
              int inIndex = inC * inW * inH + deltaY * inW + deltaX;
              float inputVal;

              if (deltaX < 0 || deltaY < 0 || deltaX >= inW || deltaY >= inH) {
                inputVal = 0.0f;
              } else {
                inputVal = input[inIndex];
              }

              int inK = outC * inChannels * kernelW * kernelH;
              inK += inC * kernelW * kernelH;
              inK += ky * kernelW;
              inK += kx;

              if (print)
                printf("%d %f %d %f\n", inIndex, inputVal, inK, kernel[inK]);

              float kernelVal = kernel[inK];
              accum += inputVal * kernelVal;
            }
          }
        }
        int outPos = outC * outW * outH;
        outPos += y * outW;
        outPos += x;

        outView[outPos] = accum;

        if (print)
          exit(0);
      }
    }
  }

  return output;
}

void *Software_Reshape(void *data, void *shape, void *output, int index,
                       ReshapeInfo *info) {
  int64_t *shapeDims = (int64_t *)shape;

  AddressGen in =
      StartAddress(info->inputDims, info->inputDims, info->numberInputDims);
  AddressGen out = StartAddress(shapeDims, shapeDims, info->numberShapeDims);

  float *inView = (float *)data;
  float *outView = (float *)output;

  // TODO: Looking at the indexes produced, it appears that the Reshape
  // operation does not need to copy data around, altought I still need to look
  // further into this.
  while (Address_IsValid(&out)) {
    int inAddr = Address_GetValue(&in);
    int outAddr = Address_GetValue(&out);

    Address_Advance(&in);
    Address_Advance(&out);

    // printf("%d %d\n",outAddr,inAddr);

    outView[outAddr] = inView[inAddr];
  }

  return output;
}

void *Software_Add(void *inputA, void *inputB, void *output, int index,
                   AddInfo *info) {
  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *out = (float *)output;

  // TODO: Instead of stuffing addressGen with the broadcasting stuff, we could
  // just iterate the broadcasted shape and call a function that would return
  // the proper index for the non broadcasted shape.
  //       Basically pull out all the properDim logic from address gen into a
  //       extra function.

  AddressGen inA =
      StartAddress(info->broadCastedShape, info->firstInputDim, info->maxDims);
  AddressGen inB =
      StartAddress(info->broadCastedShape, info->secondInputDim, info->maxDims);
  AddressGen outGen = StartAddress(info->broadCastedShape,
                                   info->broadCastedShape, info->maxDims);

  while (Address_IsValid(&outGen)) {
    int indexA = Address_GetValue(&inA);
    int indexB = Address_GetValue(&inB);
    int indexO = Address_GetValue(&outGen);

    // printf("%d %d %d\n",indexA,indexB,indexO);

    Address_Advance(&inA);
    Address_Advance(&inB);
    Address_Advance(&outGen);

    float valA = viewA[indexA];
    float valB = viewB[indexB];

    out[indexO] = valA + valB;
    // printf("%f   %f   %f\n",viewA[indexA],viewB[indexB],out[indexO]);
  }

  return output;
}

void *Software_Relu(void *inputX, void *output, int index, ReluInfo *info) {
  float *view = (float *)inputX;
  float *out = (float *)output;

  int64_t totalSize = CalculateSizeOfDim(info->inputDims, info->dims);

  for (int64_t i = 0; i < totalSize; i++) {
    float val = view[i];
    out[i] = MAX(0.0f, val);
  }

  return output;
}

void *Software_MaxPool(void *inputX, void *output, int index,
                       MaxPoolInfo *info) {
  float *view = (float *)inputX;
  float *out = (float *)output;

  AddressGen outGenInst =
      StartAddress(info->outputDims, info->outputDims, info->dims);
  AddressGen *outGen = &outGenInst;

  // We probably want to move this to the array generated by python.
  // Emit the proper array directly
  int stride[16];
  stride[0] = 1;
  stride[1] = 1;
  for (int i = 0; i < info->strideSize; i++) {
    stride[i + 2] = info->strideDims[i];
  }

  for (; Address_IsValid(outGen); Address_Advance(outGen)) {
    float max = 0.0f;
    bool firstSet = false;

    AddressGen inputPos = Address_Map(outGen, info->inputDims, stride);
    KernelGen kernInst =
        StartKernel(&inputPos, info->kernelDims, info->kernelSize);
    KernelGen *kern = &kernInst;
    for (; Kernel_IsValid(kern); Kernel_Advance(kern)) {
      if (Kernel_IsInsidePad(kern)) {
        continue;
      }
      int index = Kernel_GetValue(kern);

      float val = view[index];

      // NOTE: Padding should never affect the output value.
      //       Because negative values exist, we cannot just use zero to
      //       represent a padded value. We might get away with using -inf, but
      //       for now we just use an extra flag to check validity.
      if (!firstSet) {
        max = val;
        firstSet = true;
      } else {
        max = MAX(max, val);
      }
    }

    int outputIndex = Address_GetValue(outGen);
    out[outputIndex] = max;
  }

  return output;
}

void *Software_MatMul(void *inputA, void *inputB, void *output, int index,
                      MatMulInfo *info) {
  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *viewOut = (float *)output;

  for (int y = 0; y < info->outputDims[0]; y++) {
    for (int x = 0; x < info->outputDims[1]; x++) {
      int indexOut = info->outputDims[0] * y + x;

      viewOut[indexOut] = 0.0f;
      for (int c = 0; c < 256;
           c++) { // 256 is being hardcoded for the mnist example.
        int indexA = info->inputADims[1] * y + c;
        int indexB = info->inputBDims[1] * c + x;

        float valA = viewA[indexA];
        float valB = viewB[indexB];

        viewOut[indexOut] += valA * valB;
      }
    }
  }

  return output;
}

static inline float absf(float a) {
  if (a < 0.0f) {
    return -a;
  }
  return a;
}

int64_t CalculateSizeOfDim(int64_t *dim, int dims) {
  int64_t size = 1;
  for (int i = 0; i < dims; i++) {
    size *= dim[i];
  }

  return size;
}

void AssertAlmostEqual(void *toTest, void *correctValues, int index) {
  float *test = (float *)toTest;
  float *correct = (float *)correctValues;

  size_t outputSize = layers[index].outputSize / sizeof(float);

  if (outputSize == 0) {
    printf("Error, AssertAlmostEqual with output size of 0. Should not be "
           "possible\n");
    return;
  }

  int maxIncorrect = 10;
  bool printOk = true;

  // Make sure that cache is not affecting the verification process
  clear_cache();

  int incorrectFound = 0;
  for (int i = 0; i < outputSize; i++) {
    if (absf(correct[i] - test[i]) > 0.001f) {
      if (incorrectFound == 0) {
        printf("\n");
        printf("[%s] (Layer %d) FAIL:\n", layers[index].typeName, index);
      }
      printf("  Index: %4d Different values %.4f %.4f\n", i, correct[i],
             test[i]);
      incorrectFound += 1;
    }

    if (incorrectFound >= maxIncorrect) {
      printf("More than %d incorrect found, quitting early\n", maxIncorrect);
      printf("\n");
      break;
    }
  }

  if (printOk && incorrectFound == 0) {
    printf("[%s] (Layer %d) - OK\n", layers[index].typeName, index);
  }
}

InferenceOutput RunInference(void *outputMemory, void *temporaryMemory,
                             void **input, void *modelMemory) {
  return (InferenceOutput){};
}
