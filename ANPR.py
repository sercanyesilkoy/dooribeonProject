import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
plt.style.use('dark_background')

def anpr(img):
    # Read Input Image
    img_ori = cv2.imread(img)

    height, width, channel = img_ori.shape

    plt.figure(figsize=(12,10))
    plt.imshow(img_ori, cmap='gray')

    # Convert Image to Grayscale

    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')

    # Adaptive Thresholding

    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )

    plt.figure(figsize=(12, 10))
    plt.imshow(img_thresh, cmap='gray')

    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result)

    # Prepare Data

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        
        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')

    # Select Candidates by Char Size

    # 번호판의 문자가 될수 있는 rectangle 후보군의 조건
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0 # 번호판의 글자는 납작하기보단 길쭉하므로 비율을 이렇게 지정함
    
    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
            
    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')

    # Select Candidates by Arrangement of Contours

    # 번호판이 될 수 있는 후보들의 관계 조건
    MAX_DIAG_MULTIPLYER = 5 # rectangle 사이의 거리
    MAX_ANGLE_DIFF = 12.0 # recteangle 간의 각도
    MAX_AREA_DIFF = 0.5 # rectangle 간의 넓이 차
    MAX_WIDTH_DIFF = 0.8 # rectangle 간의 너비 차
    MAX_HEIGHT_DIFF = 0.2 # rectangle 간의 높이 차
    MIN_N_MATCHED = 3 # rectangle 집합합의 최소 갯수

    def find_chars(contour_list):
        matched_result_idx = []
        
        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            # append this contour
            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            
            # recursive
            recursive_contour_list = find_chars(unmatched_contour)
            
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx
        
    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')

    # Rotate Plate Images

    PLATE_WIDTH_PADDING = 1.3 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 2
    MAX_PLATE_RATIO = 15

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
        
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
        
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] > MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    # Another Thresholding to Find Chars

    longest_idx, longest_text = -1, 0
    plate_chars = []
    recognition = False
    result =''
    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            area = w * h
            ratio = w / h

            if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h
                    
        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
        
        result_chars = ''
        for c in chars:
            # 한국 번호판에 들어갈 수 있는 문자들 및 숫자가 인식되었을때만 결과로 출력
            if (ord('가') == ord(c) or ord('나') == ord(c) or ord('다') == ord(c) or ord('라') == ord(c) or ord('마') == ord(c) or 
                ord('거') == ord(c) or ord('너') == ord(c) or ord('더') == ord(c) or ord('러') == ord(c) or ord('머') == ord(c) or
                ord('버') == ord(c) or ord('서') == ord(c) or ord('어') == ord(c) or ord('저') == ord(c) or ord('고') == ord(c) or
                ord('노') == ord(c) or ord('도') == ord(c) or ord('로') == ord(c) or ord('모') == ord(c) or ord('보') == ord(c) or
                ord('소') == ord(c) or ord('오') == ord(c) or ord('조') == ord(c) or ord('구') == ord(c) or ord('누') == ord(c) or
                ord('두') == ord(c) or ord('루') == ord(c) or ord('무') == ord(c) or ord('부') == ord(c) or ord('수') == ord(c) or
                ord('우') == ord(c) or ord('주') == ord(c) or ord('아') == ord(c) or ord('바') == ord(c) or ord('사') == ord(c) or
                ord('자') == ord(c) or ord('배') == ord(c) or ord('허') == ord(c) or ord('하') == ord(c) or ord('호') == ord(c) or
                c.isdigit()): 
                result_chars += c

        # 인식된 글자가 4자리 이상일때만 번호판으로 인정
        if (len(result_chars) >= 4) :
            recognition = True
            result = result_chars
        print(result_chars)
        #plt.subplot(len(plate_imgs), 1, i+1)
        plt.imshow(img_result, cmap='gray')

    return recognition, result
