#include "versat_private.h"

#include "stdbool.h"
#include "stdint.h"
#include "stdlib.h" // REMOVE THIS AFTER REMOVING MALLOC AND FREE

#include "versat_accel.h"

#define MIN(A, B) (((A < B) ? (A) : (B)))
#define MAX(A, B) (((A > B) ? (A) : (B)))

void silent_clear_cache();

typedef union {
  iptr i;
  float f;
} Convertor;

iptr NoConvert(float f) {
  Convertor c = {};
  c.f = f;
  return c.i;
}

// ======================================
// Arena stuff, which we will eventually remove since we do not want to allocate
// memory while performing inference.

unsigned int Align8(unsigned int in) { return ((in + 7) & ~7); }

#define ALIGN(IN,ALIGNMENT) (((IN) + ((ALIGNMENT) - 1)) & ~((ALIGNMENT) - 1))

typedef struct Arena_t {
  void *mem;
  int used;
  int allocated;
} Arena;

void *PushBytes(Arena *arena, int size, int alignment) {
  arena->used = ALIGN(arena->used,alignment);

  int totalSize = Align8(size); // All allocates are 8 byte aligned. Do not
                                // worry about alignment further

  if (arena->used + size >= arena->allocated) {
    for (int i = 0; i < 5; i++) {
      versat_printf("\n\nArena overflow\n\n");
    }
  }

  void *res = (void *)(((char *)arena->mem) + arena->used);
  arena->used += size;

  return res;
}

#define PushType(ARENA, TYPE) (TYPE *) PushBytes(ARENA, sizeof(TYPE),__alignof__(TYPE))
#define PushArray(ARENA, COUNT, TYPE) \
  (TYPE *)PushBytes(ARENA, (COUNT) * sizeof(TYPE),__alignof__(TYPE))

typedef struct {
  unsigned int firstMarker;
  unsigned int *secondMarkerPtr;
} CanaryHeader;

bool CheckCanary(void *memory,int line) {
  CanaryHeader *asHeader = ((CanaryHeader *)memory) - 1;

  if (asHeader->firstMarker != 0x12345678) {
    versat_printf("Canary check failed before at line: %d value: %08x \n",line,asHeader->firstMarker);
    versat_printf("HeaderPtr: %p FirstMarkerPtr: %p\n",asHeader,&asHeader->firstMarker);
    return false;
  }

  if (*asHeader->secondMarkerPtr != 0x87654321) {
    versat_printf("Canary check failed after at line: %d value: %08x\n",line,*asHeader->secondMarkerPtr);
    versat_printf("SecondMarkerPtr: %p\n",asHeader->secondMarkerPtr);
    return false;
  }

  return true;
}

void *PushBytesWithCanary(Arena *arena, int size) {
  CanaryHeader *header = PushType(arena, CanaryHeader);
  void *memory = PushBytes(arena, size, 1);
  unsigned int *last = PushType(arena, unsigned int);

  header->firstMarker = 0x12345678;
  header->secondMarkerPtr = last;

  *last = 0x87654321;
  
  // This should never fail, right?
  if(!CheckCanary(memory,-1)){
    versat_printf("%p %p %p\n",header,memory,last);
  }

  return memory;
}

#define PushTypeWithCanary(ARENA, TYPE)                                        \
  (TYPE *)PushBytesWithCanary(ARENA, sizeof(TYPE))
#define PushArrayWithCanary(ARENA, COUNT, TYPE)                                \
  (TYPE *)PushBytesWithCanary(ARENA, (COUNT) * sizeof(TYPE))

typedef struct {
  Arena *arena;
  int used;
} ArenaMark;

ArenaMark MarkArena(Arena *arena) {
  ArenaMark mark = {};
  mark.arena = arena;
  mark.used = arena->used;
  return mark;
}

void MarkPop(ArenaMark mark) { mark.arena->used = mark.used; }

// TODO: All this memory allocation is very bad. We do not want to allocate
// memory at all, and we would like to push as much memory stuff to the outside
// code. We want memory to be allocated once by the user before calling
Tensor PushTensor(Arena *arena, int64_t *dims, int numberDims) {
  Tensor tensor = CreateTensor_NoAllocate(dims, numberDims);

  int size = 1;
  for (int i = 0; i < numberDims; i++) {
    size *= dims[i];
  }

  tensor.data = PushArrayWithCanary(arena, size, float);
  return tensor;
}

static void Tensor_CheckCanary_(Tensor in,int line) { CheckCanary(in.data,line); }
#define Tensor_CheckCanary(IN) Tensor_CheckCanary_(IN,__LINE__) 

Tensor Tensor_Transpose(Tensor input, int *transposeIndex, Arena *arenaOut) {
  int size = input.dims.size;
  int64_t *inDims = input.dims.data;

  int64_t outDims[MAX_DIMS] = {};

  for (int i = 0; i < size; i++) {
    int index = transposeIndex[i];
    outDims[i] = inDims[index];
  }

  Tensor res = PushTensor(arenaOut, outDims, size);

  AddressGen in = StartAddress(inDims, inDims, size);
  AddressGen out = StartAddress(outDims, outDims, size);

  for (; Address_IsValid(&in); Address_Advance(&in)) {
    int inAddr = Address_GetValue(&in);

    for (int i = 0; i < size; i++) {
      int index = transposeIndex[i];
      out.addressVars[i] = in.addressVars[index];
    }

    int outAddr = Address_GetValue(&out);

    res.data[outAddr] = input.data[inAddr];
  }

  return res;
}

Tensor Tensor_ExtractView(Tensor input, int dimIndex, int start, int size,
                          Arena *arenaOut) {
  AddressGen in =
      StartAddress(input.dims.data, input.dims.data, input.dims.size);

  in.offsetAddressVars[dimIndex] = start;
  in.iterationDims[dimIndex] = (int64_t)size;

  Dimensions outDims = input.dims;
  outDims.data[dimIndex] = size;

  Tensor output = PushTensor(arenaOut, outDims.data, outDims.size);
  AddressGen out = StartAddress(outDims.data, outDims.data, outDims.size);

  for (; Address_IsValid(&in); Address_Advance(&in), Address_Advance(&out)) {
    int inIndex = Address_GetValue(&in);
    int outIndex = Address_GetValue(&out);

    output.data[outIndex] = input.data[inIndex];
  }

  return output;
}

static inline int64_t GetDim(int64_t *dimArray, int dimSize, int index) {
  if (index < dimSize) {
    return MAX(dimArray[index], 1);
  }

  return 1;
}

static inline int64_t GetSize(int64_t *dimArray, int dimSize, int index) {
  if (index < dimSize) {
    if (dimArray[index] > 1) {
      return 1;
    }
  }

  return 0;
}

#define SWAP(TYPE, A, B)                                                       \
  do {                                                                         \
    TYPE t = A;                                                                \
    A = B;                                                                     \
    B = t;                                                                     \
  } while (0)

#include <string.h>

void *Versat_Add(void *inputA, void *inputB, void *output, int index,
                 AddInfo *info) {
  int64_t *l = info->firstInputDim;
  int64_t *r = info->secondInputDim;
  int64_t *o = info->broadCastedShape;
  int d = info->maxDims;

  Dimensions left = CreateDimensions(l, d);
  Dimensions right = CreateDimensions(r, d);

  if (Dimensions_TotalSize(left) < Dimensions_TotalSize(right)) {
    SWAP(void *, inputA, inputB);
    SWAP(Dimensions, left, right);
    SWAP(int64_t *, l, r);
  }

  volatile Top_AddConfig *config = &accelConfig->Top_Add;
  ActivateMergedAccelerator(MergeType_Top_Add);

  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *out = (float *)output;

  // TODO: Eventually this will depend on the size of Versat memories
  //       The bigger the more efficient we can be
  int maxLineSupported = 1024;

  AddressGen inA = StartAddress(o, l, info->maxDims);
  AddressGen inB = StartAddress(o, r, info->maxDims);
  AddressGen outGen = StartAddress(o, o, info->maxDims);

  int lineLength = l[d - 1];
  while (Address_IsValid(&outGen)) {
    int indexA = Address_GetValue(&inA);
    int indexB = Address_GetValue(&inB);
    int indexO = Address_GetValue(&outGen);

    bool broadcastedB = (GetSize(r, d, d - 1) == 0);

    for (int offset = 0; offset < lineLength; offset += maxLineSupported) {
      int trueLength = MIN(maxLineSupported, lineLength - offset);

      Top_Add_Linear(&viewA[indexA + offset], trueLength);
      Top_Add_Broadcast(&viewB[indexB + (broadcastedB ? 0 : offset)],
                        trueLength, GetSize(r, d, d - 1));
      Top_Add_Output(&out[indexO + offset], trueLength);

      RunAccelerator(1);
    }

    for (int i = 0; i < lineLength; i++) {
      Address_Advance(&inA);
      Address_Advance(&inB);
      Address_Advance(&outGen);
    }
  }

  config->inputs_0.enabled = 0;
  config->inputs_1.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  return output;
}

void *Versat_Relu(void *inputA, void *output, int index, ReluInfo *info) {
  volatile Top_ReluConfig *config = &accelConfig->Top_Relu;

  ActivateMergedAccelerator(MergeType_Top_Relu);

  int dims = info->dims;
  int64_t *dim = info->inputDims;

  int64_t totalSize = CalculateSizeOfDim(dim, dims);

  int64_t maxAtATime = 256;

  float *inputView = (float *)inputA;
  float *outputView = (float *)output;

  for (int i = 0; i < totalSize; i += maxAtATime) {
    int size = MIN(maxAtATime, totalSize - i);

    Top_Relu_Simple(&inputView[i], &outputView[i], size);

    RunAccelerator(1);
  }

  config->input.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  return output;
}

void *Versat_Reshape(void *data, void *shape, void *output, int index,
                     ReshapeInfo *info) {
  return data;
}

static inline void MaxPool_ProcessWindow(AdvancedWindow w, int channel,
                                         void *input, void *output,
                                         MaxPoolInfo *info) {
  volatile Top_MaxpoolConfig *config = &accelConfig->Top_Maxpool;

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  int outputImageW = info->outputDims[3];
  int outputImageH = info->outputDims[2];

  int cInStart = channel * inputImageH * inputImageW;
  int cOutStart = channel * outputImageH * outputImageW;

  int stride = w.actualKernelW * w.actualKernelH;

  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  Top_Maxpool_Features(input, w.inputX, w.inputY, cInStart, w.actualKernelW,
                       w.actualKernelH, inputImageW, w.outputW, w.outputH,
                       strideW, strideH);

  Top_Maxpool_Output(output, w.outputX, w.outputY, cOutStart, w.outputW,
                     w.outputH, outputImageW, stride);

  config->accum.strideMinusOne = stride - 1;
  StartAccelerator();
}

// Currently hardcoded for 2D kernels.
void *Versat_MaxPool(void *inputX, void *output, int index, MaxPoolInfo *info) {
  forceDoubleLoop = true;
  volatile Top_MaxpoolConfig *config = &accelConfig->Top_Maxpool;
  ActivateMergedAccelerator(MergeType_Top_Maxpool);

  int channels = info->inputDims[1];

  ExtraInfo extra = CalculateExtraInfo_MaxPool(info);

  // MaxPool is currently using NCHW. We iterate by channels since there is no
  // gain in passing a window that spans channels.

  // For MaxPool using NHWC the approach might be different.
  // We might want to use windows that span channels
  for (int c = 0; c < channels; c++) {
    WindowGen genInst = StartWindowGen(&extra, false, false);
    WindowGen *gen = &genInst;

    for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
      AdvancedWindow w = WindowGen_Get(gen);
      MaxPool_ProcessWindow(w, c, inputX, output, info);
    }
  }

  // Flush the remaining data from the accelerator
  config->features.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  return output;
}

static inline void AveragePool_ProcessWindow(AdvancedWindow w, int channel,
                                             void *input, void *output,
                                             AveragePoolInfo *info) {
  volatile Top_AveragePoolConfig *config = &accelConfig->Top_AveragePool;

  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  int outputImageW = info->outputDims[3];
  int outputImageH = info->outputDims[2];

  int cInStart = channel * inputImageH * inputImageW;
  int cOutStart = channel * outputImageH * outputImageW;

  int stride = w.actualKernelW * w.actualKernelH;

  int strideW = info->strideDims[1];
  int strideH = info->strideDims[0];

  Top_AveragePool_Features(input, w.inputX, w.inputY, cInStart, w.actualKernelW,
                           w.actualKernelH, inputImageW, w.outputW, w.outputH,
                           strideW, strideH);
  Top_AveragePool_Output(output, w.outputX, w.outputY, cOutStart, w.outputW,
                         w.outputH, outputImageW, stride);

  config->averagePool_accum.strideMinusOne = stride - 1;
  config->invertedDivisor.constant = NoConvert(1.0f / (float)stride);
  StartAccelerator();
}

// Currently hardcoded for 2D kernels.
void *Versat_AveragePool(void *inputX, void *output, int index,
                         AveragePoolInfo *info) {
  forceDoubleLoop = true;
  volatile Top_AveragePoolConfig *config = &accelConfig->Top_AveragePool;
  ActivateMergedAccelerator(MergeType_Top_AveragePool);

  int channels = info->inputDims[1];

  ExtraInfo extra = CalculateExtraInfo_AveragePool(info);

  // Using NHWC
  for (int c = 0; c < channels; c++) {
    WindowGen genInst = StartWindowGen(&extra, false, false);
    WindowGen *gen = &genInst;

    for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
      AdvancedWindow w = WindowGen_Get(gen);
      AveragePool_ProcessWindow(w, c, inputX, output, info);
    }
  }

  // Flush the remaining data from the accelerator
  config->features.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  return output;
}

void ConvWithBias_ProcessWindow(AdvancedWindow w, void *inputX, void *inputW,
                                void *outAddr, float *bias, ConvInfo *info,
                                int inputC, int outputC) {
  volatile Top_ConvConfig *config = &accelConfig->Top_Conv;

  int inputImageW = info->inputDims[3];
  int inputImageC = inputC;

  int outputImageW = info->outputDims[3];
  int outputImageC = outputC;

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  int stride = w.actualKernelW * w.actualKernelH * inputImageC;

  int convChannelSize = inputImageC;
  int group = info->group;

  int convStartC = 0; // We must always process the entire input channels.

#if 0
  static unsigned int delayBuffer[] = {0x0, 0x0, 0x10, 0x0, 0x0, 0x6};
  ;
  VersatLoadDelay(delayBuffer);
#endif

  Top_Conv_FeaturesWeightsOutputs(
      inputX, inputW, outAddr, w.actualKernelW, w.actualKernelH,
      convChannelSize,

      w.outputH, w.outputW, w.outputSizeC,

      w.inputX, w.inputY, w.kernelStartW, w.kernelStartH, w.startC, w.outputX,
      w.outputY,

      inputImageW, inputImageC, convStartC, kernelW, kernelH,

      outputImageW, stride, outputC);

  if (bias == NULL) {
    static float bias = 0.0f;
    Top_Conv_Bias(&bias, 1, 1);
  } else {
    Top_Conv_Bias(bias + w.startC, w.outputSizeC, stride);
  }

  config->myAccum.strideMinusOne = stride - 1;

  StartAccelerator();
}

void *Versat_Conv(void *inputX, void *inputW, void *output, int index,
                  ConvInfo *info) {
  Versat_ConvWithBias(inputX, inputW, NULL, output, index, info);
}

void *Versat_ConvWithBias(void *inputX, void *inputW, void *inputB,
                          void *output, int index, ConvInfo *info) {
  forceDoubleLoop = true;

  volatile Top_ConvConfig *config = &accelConfig->Top_Conv;

  static Arena arenaInst = {};
  static Arena *arena = &arenaInst;

  if(!arena->mem){
    arena->allocated = 1024 * 1024 * 16;
    arena->mem = malloc(arena->allocated); // 16 Megabytes
  }

  ArenaMark outerMark = MarkArena(arena);

  ActivateMergedAccelerator(MergeType_Top_Conv);

  int batches = info->inputDims[0];
  int inputChannels = info->inputDims[1];
  int inputImageW = info->inputDims[3];
  int inputImageH = info->inputDims[2];

  int outputChannels = info->outputDims[1];
  int outputImageH = info->outputDims[2];
  int outputImageW = info->outputDims[3];

  int inputSize = inputImageW * inputImageH * inputChannels;
  int outputSize = outputImageW * outputImageH * outputChannels;
  int group = info->group;

  int kernelW = info->kernelDims[1];
  int kernelH = info->kernelDims[0];

  VersatVarSpec outputHSpec = {1, outputImageH, 0};
  VersatVarSpec outputWSpec = {1, outputImageW, 1};
  VersatVarSpec outputCSpec = {1, outputChannels, 2};
  int bytesUsed = Top_Conv_FeaturesWeightsOutputs_Size(
      kernelW, kernelH, inputChannels, &outputHSpec, &outputWSpec,
      &outputCSpec);

  Tensor inputTensor = CreateTensor_NoAllocate(info->inputDims, 4);
  inputTensor.data = inputX;

  for (int batch = 0; batch < batches; batch++) {
    ArenaMark mark = MarkArena(arena);

    // TODO: This technically depends on batch because we have group related
    // operations that change these values.
    // If we remove them we can then push this outside the loop
    ExtraInfo extra = CalculateExtraInfo_Conv(info);
    // ExtraInfo_Print(extra);

    int64_t NHWCDims[] = {info->inputDims[0], info->inputDims[2],
                          info->inputDims[3], info->inputDims[1]};

    Tensor tempInputTensor = PushTensor(arena, NHWCDims, 4);

    Tensor_CheckCanary(tempInputTensor);

    Tensor tempOutputTensor = PushTensor(arena, info->outputDims, 4);

    Tensor_CheckCanary(tempInputTensor);

    int64_t kernelDims[] = {outputChannels, inputChannels / group,
                            info->kernelDims[1], info->kernelDims[0]};
    // Tensor kernel = CreateTensor_NoAllocate(kernelDims, 4);
    // kernel.data = inputW;

    Tensor_CheckCanary(tempInputTensor);

    int kernelSize = Dimensions_TotalSize(CreateDimensions(kernelDims, 4));

    float *tempInput = tempInputTensor.data;
    float *tempOutput = tempOutputTensor.data;

    float *inputView = (float *)inputX;
    float *biasView = (float *)inputB;

    inputView += batch * inputSize;

    Tensor_CheckCanary(tempInputTensor);

    // Convert NCHW to NHWC
    for (int y = 0; y < inputImageH; y++) {
      for (int x = 0; x < inputImageW; x++) {
        for (int c = 0; c < inputChannels; c++) {
          int NCHW_Index =
              c * (inputImageH * inputImageW) + y * inputImageW + x;
          int NHWC_Index =
              y * (inputImageW * inputChannels) + x * inputChannels + c;

          tempInput[NHWC_Index] = inputView[NCHW_Index];
        }
      }
    }

    Tensor_CheckCanary(tempInputTensor);

    // Extract the channel
    Dimensions dims = CreateDimensions(info->inputDims, 4);
    dims.data[1] /= group;

    int size = Dimensions_TotalSize(dims);

    Dimensions outDims = CreateDimensions(info->outputDims, 4);
    outDims.data[1] /= group;

    int64_t NHWCOutDims[4] = {outDims.data[0], outDims.data[2], outDims.data[3],
                              outDims.data[1]};
    Tensor tempGroupTensor = PushTensor(arena, NHWCOutDims, 4);
    float *tempGroupOutput = tempGroupTensor.data;

    // TODO: Changing extra is kinda "problematic". We are doing a bunch of
    // stuff that might not be needed anymore.
    //       It might be possible to just push this logic to Versat and let it
    //       handle it.
    extra.inputImageC /= group;
    extra.outputImageC /= group;

    int index = 0;
    for (int g = 0; g < group; g++) {
      int inputC = extra.inputImageC;
      int outputC = extra.outputImageC;

      // We extract the input associated to the current group.
      Tensor extracted =
          Tensor_ExtractView(tempInputTensor, 3, g * inputC, inputC, arena);

      // We iterate over the "reduced" extra values.
      WindowGen genInst = StartWindowGen(&extra, true, false);
      WindowGen *gen = &genInst;

      // We extract the bias input.
      float *trueBias = biasView;
      if (trueBias != NULL) {
        trueBias += (g * extra.outputImageC);
      }

      for (; WindowGen_Valid(gen); WindowGen_Advance(gen)) {
        AdvancedWindow w = WindowGen_Get(gen);

        if (w.entireWindowInsidePadding) {
          float bias = 0.0f;
          if (trueBias) {
            bias = trueBias[w.outputC];
          }
          tempGroupOutput[w.outputY * extra.outputImageC * outputImageW +
                          w.outputX * extra.outputImageC + w.outputC] = bias;
        } else {
          ConvWithBias_ProcessWindow(
              w, extracted.data, ((float *)inputW) + g * (kernelSize / group),
              tempGroupOutput, trueBias, info, inputC, outputC);
        }
      }

      // Flush the remaining data from the accelerator
      // TODO: Not efficient but not worrying about it for now.
      config->features.enabled = 0;
      config->weights.enabled = 0;
      config->output.enabled = 0;
      config->bias.enabled = 0;
      RunAccelerator(2);

      silent_clear_cache();

      // We obtain the result in NHWC format and we need to "concatenate" this
      // with the output that we are building.
      // The output is also in NHWC format.
      // The problem is that the concatenation assumes that we are in NCHW
      // format.

      // We then concatenate everything into one place.
      // And make use of the fact that in NCHW we can just "append".
      // So it is easier to transpose the small output patch than it is to
      int transposeDims[] = {0, 3, 1, 2};
      Tensor transposed =
          Tensor_Transpose(tempGroupTensor, transposeDims, arena);

      float *outputView = (float *)output;
      outputView += batch * outputSize; // + g * (outputSize / group);
      for (int i = 0; i < outputSize / group; i++) {
        outputView[index++] = transposed.data[i];
      }

      Tensor_CheckCanary(extracted);
      Tensor_CheckCanary(transposed);
    }

    Tensor_CheckCanary(tempGroupTensor);

    Tensor_CheckCanary(tempInputTensor);
    Tensor_CheckCanary(tempOutputTensor);

    MarkPop(mark);
  }

  MarkPop(outerMark);

  return output;
}

void *Versat_MatMul(void *inputA, void *inputB, void *output, int index,
                    MatMulInfo *info) {

  ActivateMergedAccelerator(MergeType_Top_MatMul);
  volatile Top_MatMulConfig *config = &accelConfig->Top_MatMul;

  float *viewA = (float *)inputA;
  float *viewB = (float *)inputB;
  float *viewOut = (float *)output;

  // TODO: The names are kinda wrong. AH and AW are "technically" swapped in
  // name only.
  int AS = info->numberInputADims;
  int AH;
  int AW;
  if (AS == 1) {
    AH = 1;
    AW = info->inputADims[0];
  } else {
    AH = info->inputADims[AS - 2];
    AW = info->inputADims[AS - 1];
  }

  int BS = info->numberInputBDims;
  int BH;
  int BW;
  if (BS == 1) {
    BH = 1;
    BW = info->inputBDims[0];
  } else {
    BH = info->inputBDims[BS - 2];
    BW = info->inputBDims[BS - 1];
  }

  int totalBSize = BH * BW;
  float *tempB = (float *)malloc(sizeof(float) * totalBSize);

  int OS = info->numberOutputDims;
  int OH;
  int OW;
  if (OS == 1) {
    OH = 1;
    OW = info->outputDims[0];
  } else {
    OH = info->outputDims[OS - 2];
    OW = info->outputDims[OS - 1];
  }

  if (AW != BH) {
    versat_printf("Something very wrong is happening in MatMul\n");
  }

  Dimensions dimA = CreateDimensions(info->inputADims, info->numberInputADims);
  Dimensions dimB = CreateDimensions(info->inputBDims, info->numberInputBDims);
  Dimensions dimO = CreateDimensions(info->outputDims, info->numberOutputDims);

  if (dimA.size == 1) {
    Dimensions_PrependInPlace(&dimA, 1);
  }
  if (dimB.size == 1) {
    Dimensions_PrependInPlace(&dimB, 1);
  }
  if (dimO.size == 1) {
    Dimensions_PrependInPlace(&dimO, 1);
  }

  int dimsToPreserve = 2;
  int dimsToIterateA = MAX(0, dimA.size - dimsToPreserve);
  int dimsToIterateB = MAX(0, dimB.size - dimsToPreserve);
  int dimsToIterateO = MAX(0, dimO.size - dimsToPreserve);

  AddressGen addrA = StartAddressFromDims(dimA, dimsToIterateA);
  AddressGen addrB = StartAddressFromDims(dimB, dimsToIterateB);
  AddressGen addrO = StartAddressFromDims(dimO, dimsToIterateO);

  while (Address_IsValid(&addrA) || Address_IsValid(&addrB) ||
         Address_IsValid(&addrO)) {
    if (!Address_IsValid(&addrA)) {
      Address_Restart(&addrA);
    }
    if (!Address_IsValid(&addrB)) {
      Address_Restart(&addrB);
    }
    if (!Address_IsValid(&addrO)) {
      Address_Restart(&addrO);
    }

    int valA = Address_GetValue(&addrA);
    int valB = Address_GetValue(&addrB);
    int valO = Address_GetValue(&addrO);

    EndAccelerator();
    for (int y = 0; y < BH; y++) {
      for (int x = 0; x < BW; x++) {
        // Transposing B
        tempB[x * BH + y] = viewB[y * BW + x + valB];
      }
    }

    silent_clear_cache();

    for (int y = 0; y < OH; y++) {
      for (int x = 0; x < OW; x++) {
        float *lineAStart = &viewA[y * AW + valA];
        float *lineBStart = &tempB[x * AW];

        float *out = &viewOut[y * OW + x + valO];

        Top_MatMul_Simple(lineAStart, lineBStart, AW);
        Top_MatMul_Output(out, 1, AW);

        config->myAccum.strideMinusOne = AW - 1;

        StartAccelerator();
      }
    }

    Address_Advance(&addrA);
    Address_Advance(&addrB);
    Address_Advance(&addrO);
  }

  config->leftRow.enabled = 0;
  config->rightRow.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  free(tempB);

  return output;
}

void *Versat_Softmax(void *inputA, void *output, int index, SoftmaxInfo *info) {
  return Software_Softmax(inputA, output, index, info);
}

void *Versat_BatchNormalization(void *inputX, void *scale, void *inputB,
                                void *mean, void *var, void *output, int index,
                                BatchNormalizationInfo *info) {
  ActivateMergedAccelerator(MergeType_Top_BatchNormalization);

  float *x = (float *)inputX;
  float *s = (float *)scale;
  float *b = (float *)inputB;
  float *m = (float *)mean;
  float *v = (float *)var;
  float *o = (float *)output;

  Dimensions dim = CreateDimensions(info->inputDims, info->numberInputDims);

  if (dim.size <= 1) {
    Dimensions_AppendInPlace(&dim, 1);
  }

  int totalC = dim.data[1];

  float *A = (float *)malloc(sizeof(float) * totalC);
  float *B = (float *)malloc(sizeof(float) * totalC);
  for (int c = 0; c < totalC; c++) {
    float inv = my_invsqrt(v[c] + info->epsilon);
    A[c] = s[c] * inv;
    B[c] = (-m[c] * inv) * s[c] + b[c];
  }

  AddressGen addrInst = StartAddressFromDims(dim, 2);
  AddressGen *addr = &addrInst;

  // TODO: We probably can also do this using the Kernel stuff.
  //       But I kinda want a better interface when using kernel stuff.
  Dimensions leftover = Dimensions_Cut_GetRight(dim, 2);
  int size = Dimensions_TotalSize(leftover);

  VersatVarSpec sizeSpec;
  sizeSpec.min = 1;
  sizeSpec.max = size;
  int bytesTransferPerRun = Top_BatchNormalization_Simple_Size(&sizeSpec);

  int transferSize = sizeSpec.value;

  silent_clear_cache();

  while (Address_IsValid(addr)) {
    int c = Address_GetDim(addr, 1);

    int index = Address_GetValue(addr);

    union {
      int i;
      float f;
    } a, b;

    a.f = A[c];
    b.f = B[c];

    for (int i = 0; i < size; i += transferSize) {
      int trueSize = MIN(size - i, transferSize);
      Top_BatchNormalization_Simple(x, o, index + i, trueSize, a.i, b.i);
      StartAccelerator();
    }

    Address_Advance(addr);
  }

  EndAccelerator();

  volatile Top_BatchNormalizationConfig *config =
      &accelConfig->Top_BatchNormalization;
  config->x.enabled = 0;
  config->o.enabled = 0;
  RunAccelerator(2);

  free(A);
  free(B);

  return o;
}

void *Versat_Dropout(void *input, void *out, int index, DropoutInfo *info) {
  Tensor asTensor =
      CreateTensor_NoAllocate(info->inputDims, info->numberInputDims);
  int size = Tensor_Size(asTensor);

  float *asFloatIn = (float *)input;
  float *asFloatOut = (float *)out;

  for (int i = 0; i < size; i++) {
    asFloatOut[i] = asFloatIn[i];
  }

  return input;
}

void *Versat_LRN(void *input, void *out, int index, LRNInfo *info) {
  return Software_LRN(input, out, index, info);
}

void *Versat_Gemm(void *inA, void *inB, void *inC, void *out, int index,
                  GemmInfo *info) {
  ActivateMergedAccelerator(MergeType_Top_Gemm);
  volatile Top_GemmConfig *config = &accelConfig->Top_Gemm;

  float *viewA = (float *)inA;
  float *viewB = (float *)inB;
  float *viewC = (float *)inC;
  float *viewOut = (float *)out;

  int AH = info->aDims[0]; // 1
  int AW = info->aDims[1]; // 4

  int totalASize = AH * AW;
  float *tempA = (float *)malloc(sizeof(float) * totalASize);

  int BH = info->bDims[0];
  int BW = info->bDims[1];

  int totalBSize = BH * BW;
  float *tempB = (float *)malloc(sizeof(float) * totalBSize);

  int CH = info->cDims[0];
  int CW = info->cDims[1];

  int trueAW = AW;
  int trueAH = AH;
  if (info->transA) {
    trueAW = AH;
    trueAH = AW;
  }

  int trueBW = BW;
  int trueBH = BH;
  if (info->transB) {
    trueBW = BH;
    trueBH = BW;
  }

  int OH = trueAH;
  int OW = trueBW;

  int broadCastH = (OH == CH && OH != 1) ? 1 : 0;
  int broadCastW = (OW == CW && OW != 1) ? 1 : 0;

  int64_t dimsOut[2] = {OH, OW};

  Dimensions dimA = CreateDimensions(info->aDims, info->numberDims);
  Dimensions dimB = CreateDimensions(info->bDims, info->numberDims);
  Dimensions dimC = CreateDimensions(info->cDims, info->numberDims);

  Dimensions dimO = CreateDimensions(dimsOut, info->numberDims);

  AddressGen addrA = StartAddressFromDims(dimA, 0);
  AddressGen addrB = StartAddressFromDims(dimB, 0);
  AddressGen addrO = StartAddressFromDims(dimO, 0);

#if 0
  while (Address_IsValid(&addrA) || Address_IsValid(&addrB) ||
         Address_IsValid(&addrO)) {
    int valA = Address_GetValue(&addrA);
    int valB = Address_GetValue(&addrB);
    int valO = Address_GetValue(&addrO);

    EndAccelerator();
#endif
  int valA = 0;
  int valB = 0;
  int valO = 0;

  // By default we transpose B in order to implement the multiplication phase
  // directly. Which means that we do the opposite when we want to "transpose"
  // B.
  float *properBInput = viewB;
  if (!info->transB) {
    for (int y = 0; y < BH; y++) {
      for (int x = 0; x < BW; x++) {
        // Transposing B
        tempB[x * BH + y] = viewB[y * BW + x + valB];
      }
    }

    properBInput = tempB;
  }

  // versat_printf("AW: %d,AH: %d,BW: %d,BH: %d,OW: %d,OH:
  // %d\n",AW,AH,BW,BH,OW,OH);

  Top_Gemm_Alpha(NoConvert(info->alpha));

  for (int y = 0; y < OH; y++) {

    float *properAInput = &viewA[y * AW];
    if (info->transA) {
      for (int x = 0; x < AH; x++) {
        // Transposing A
        tempA[x] = viewA[x * AW + y];
      }
      properAInput = tempA;
    }

    silent_clear_cache();

    for (int x = 0; x < OW; x++) {
      float *lineAStart = properAInput;
      float *lineBStart = &properBInput[x * trueAW];

      float *out = &viewOut[y * OW + x + valO];

      int cIndex = y * (broadCastH ? CW : 0) + x * (broadCastW ? 1 : 0);

      float cVal = viewC[cIndex];

      Top_Gemm_CValue(NoConvert(cVal * info->beta));

      Top_Gemm_Simple(lineAStart, lineBStart, trueAW);
      Top_Gemm_Output(out, 1, trueAW);

      config->myAccum.strideMinusOne = trueAW - 1;

      StartAccelerator();
    }
  }

#if 0
    Address_Advance(&addrA);
    Address_Advance(&addrB);
    Address_Advance(&addrO);
  }
#endif

  config->gemmLeftRow.enabled = 0;
  config->gemmRightRow.enabled = 0;
  config->output.enabled = 0;
  RunAccelerator(2);

  free(tempA);
  free(tempB);

  return viewOut;
}
