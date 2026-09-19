"""COCO class names and their Arabic translations."""

# COCO 80-class list indexed by the 91-id COCO category ids that RF-DETR emits.
# Used only as a fallback if `rfdetr.util.coco_classes` is unavailable.
COCO_ID_TO_NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard",
    42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}

# English COCO label -> Arabic. Anything missing falls back to the English label.
AR = {
    "person": "شخص",
    "chair": "كرسي",
    "cup": "كوب",
    "laptop": "حاسوب",
    "cell phone": "هاتف",
    "bottle": "زجاجة",
    "book": "كتاب",
    "tv": "تلفاز",
    "keyboard": "لوحة مفاتيح",
    "mouse": "فأرة",
    "car": "سيارة",
    "backpack": "حقيبة",
    "handbag": "حقيبة يد",
    # extras that show up constantly on a desk / in a room
    "dining table": "طاولة",
    "couch": "أريكة",
    "bed": "سرير",
    "potted plant": "نبتة",
    "remote": "جهاز تحكم",
    "clock": "ساعة",
    "vase": "مزهرية",
    "bowl": "وعاء",
    "wine glass": "كأس",
    "fork": "شوكة",
    "knife": "سكين",
    "spoon": "ملعقة",
    "scissors": "مقص",
    "umbrella": "مظلة",
    "tie": "ربطة عنق",
    "suitcase": "حقيبة سفر",
    "bicycle": "دراجة",
    "motorcycle": "دراجة نارية",
    "bus": "حافلة",
    "truck": "شاحنة",
    "dog": "كلب",
    "cat": "قطة",
    "bird": "طائر",
    "banana": "موزة",
    "apple": "تفاحة",
    "orange": "برتقالة",
    "pizza": "بيتزا",
    "cake": "كعكة",
    "sink": "حوض",
    "refrigerator": "ثلاجة",
    "microwave": "ميكروويف",
    "oven": "فرن",
    "toothbrush": "فرشاة أسنان",
    "teddy bear": "دمية دب",
    "sports ball": "كرة",
    "skateboard": "لوح تزلج",
    "traffic light": "إشارة مرور",
}


def arabic(name: str) -> str:
    """Arabic label for a COCO class, falling back to the English name."""
    return AR.get(name, name)
