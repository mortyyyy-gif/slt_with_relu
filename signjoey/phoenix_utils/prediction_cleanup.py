from itertools import groupby
import re

def clean_prediction(prediction):

    prediction = prediction.strip()

    # حذف فاصله‌های اضافی
    prediction = re.sub(r"\s+", " ", prediction)

    # حذف تکرارهای پشت سر هم
    prediction = " ".join(
        " ".join(i[0] for i in groupby(prediction.split()))
        .split()
    )

    return prediction.strip()