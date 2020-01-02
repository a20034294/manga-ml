import manga109api
from pprint import pprint
import cv2
import time
import os
from dotenv import load_dotenv

load_dotenv()

manga109_root_dir = os.getenv('MANGA109_ROOT_DIR')
p = manga109api.Parser(root_dir=manga109_root_dir)

book_cnt = 0
os.mkdir('face/')

for book in p.books:
    book_cnt += 1
    face_cnt = 0
    os.mkdir('face/' + str(book_cnt))
    for page in p.annotations[book]["book"]["pages"]["page"]:
        img = cv2.imread(p.img_path(book=book, index=page["@index"]))
        if "face" in page.keys() and type(page["face"]) is list:
            for face in page["face"]:
                #pprint(face)
                x, y = face["@xmin"], face["@ymin"]
                w = face["@xmax"] - x
                h = face["@ymax"] - y
                if w > h:
                    offset = int((w - h) / 2)
                    face_img = img[y:y + h, x + offset:x + h + offset]
                else:
                    offset = int((h - w) / 2)
                    face_img = img[y + offset:y + w + offset, x:x + w]
            if face_img.shape[0] >= 50:
                ok_img = cv2.resize(
                    face_img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
                face_cnt += 1
                cv2.imwrite(
                    'face/' + str(book_cnt) + '/' + str(face_cnt) + '.jpg',
                    ok_img)
        if "face" in page.keys() and type(page["face"]) is dict:
            face = page["face"]
            x, y = face["@xmin"], face["@ymin"]
            w = face["@xmax"] - x
            h = face["@ymax"] - y
            if w > h:
                offset = int((w - h) / 2)
                face_img = img[y:y + h, x + offset:x + h + offset]
            else:
                offset = int((h - w) / 2)
                face_img = img[y + offset:y + w + offset, x:x + w]
            if face_img.shape[0] >= 50:
                ok_img = cv2.resize(
                    face_img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
                face_cnt += 1
                cv2.imwrite(
                    'face/' + str(book_cnt) + '/' + str(face_cnt) + '.jpg',
                    ok_img)
