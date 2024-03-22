from enum import Enum, auto

class DType(Enum):
    float32 = auto()
    float = auto()
    float64 = auto()
    double = auto()
    float16 = auto()
    bfloat16 = auto()
    half = auto()
    uint8 = auto()
    int8 = auto()
    int16 = auto()
    short = auto()
    int32 = auto()
    int = auto()
    int64 = auto()
    long = auto()
    complex32 = auto()
    complex64 = auto()
    cfloat = auto()
    complex128 = auto()
    cdouble = auto()
    quint8 = auto()
    qint8 = auto()
    qint32 = auto()
    bool = auto()
    quint4x2 = auto()
    quint2x4 = auto()
    
    @staticmethod
    def sizeof(t):
        if t == DType.float16 or t == DType.half:
            return 2
        elif t == DType.float32 or t == DType.float:
            return 4
        elif t == DType.float64 or t == DType.double:
            return 8
        elif t == DType.int8 or t == DType.uint8 or t == DType.quint8 or t == DType.qint8:
            return 1
        elif t == DType.int16 or t == DType.short:
            return 2
        elif t == DType.int32 or t == DType.int:
            return 4
        elif t == DType.int64 or t == DType.long:
            return 8
        else:
            raise Exception(f'unknown dtype {t}')