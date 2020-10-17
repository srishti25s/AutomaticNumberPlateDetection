def start_video(fname):
    import socket

    HOST = '127.0.0.1'
    PORT = 2001
    import cv2
    import numpy as np
    import requests
    from datetime import datetime
    import os
    import base64

    from configparser import ConfigParser
    import psycopg2
    from config_db import config

    if not os.path.exists('static/plate_result'):
        os.mkdir('static/plate_result')

    if not os.path.exists('static/vehicle_result'):
        os.mkdir('static/vehicle_result')

    plate_weights_path = "./model/plate.weights"
    plate_cfg_path = "./model/plate.cfg"
    plate_classes_path = "./model/plate.txt"
    number_weights_path = "./model/recog.weights"
    number_cfg_path = "./model/recog.cfg"
    number_classes_path = "./model/recog.txt"
    color_weights_path = "./model/color.weights"
    color_cfg_path = "./model/color.cfg"
    color_classes_path = "./model/color.txt"

    plate_classes = None
    number_classes = None
    plate_net = None
    number_net = None
    COLORS = None
    connection = None
    camera_id = None
    temp_license_str = ""
    CamName = None
    frame_origin = None
    data_info = []
    previous_license = ""

    global image_width, image_height,temp_result

    temp_result = ""
    image_width = 0
    image_height = 0

    def get_param(name):
        import ast 
        config = ConfigParser()
        config.read('config_param.ini')
        obj = config.get('main',name)
        obj = ast.literal_eval(obj)
        return obj


    with open(plate_classes_path, 'r') as f:
        plate_classes = [line.strip() for line in f.readlines()]
    with open(number_classes_path, 'r') as f:
        number_classes = [line.strip() for line in f.readlines()]
    with open(color_classes_path, 'r') as f:
        color_classes = [line.strip() for line in f.readlines()]
        
    plate_net = cv2.dnn.readNet(plate_weights_path, plate_cfg_path)
    number_net = cv2.dnn.readNet(number_weights_path, number_cfg_path)
    color_net = cv2.dnn.readNet(color_weights_path, color_cfg_path)
    COLORS = np.random.uniform(0, 255, size=(len(number_classes), 3))

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_box( img, class_id, label, x, y, x_plus_w, y_plus_h):
        # label = str(plate_classes[class_id])
        color = (0, 255, 0)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_plate(image, plate_type):
        scale = 0.00392

        # read pre-trained model and config file

        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (200, 80), (0, 0, 0), True, crop=False)

        # set input blob for the network
        number_net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = number_net.forward(get_output_layers(number_net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        center_X = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    center_X.append(center_x)

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box

        result = ''

        valid_boxes = []
        valid_classids = []
        valid_centerX = []
        valid_centerY = []
        valid_centerH = []

        valid_topY = []
        lineFlag = False
        average_heightY = 0
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            valid_boxes.append(box)
            valid_classids.append(class_ids[i])
            valid_centerX.append(round(x))
            valid_centerY.append(round(y))
            valid_centerH.append(round(w))
            draw_bounding_box(image, class_ids[i], "", round(x), round(y), round(x + w), round(y + h))
            average_heightY = average_heightY + round(y + h / 2)

        if (len(valid_centerY) > 0):
            average_heightY = average_heightY / len(valid_centerY)

        for i in range(0, len(valid_centerX)):
            if (valid_centerY[i] > average_heightY):
                lineFlag = True
                break
        if (lineFlag == False):
            for i in range(0, len(valid_centerX)):
                for j in range(i + 1, len(valid_centerX)):
                    if valid_centerX[i] > valid_centerX[j]:
                        temp = valid_centerX[i]
                        valid_centerX[i] = valid_centerX[j]
                        valid_centerX[j] = temp

                        tem = valid_classids[i]
                        valid_classids[i] = valid_classids[j]
                        valid_classids[j] = tem
            for i in range(0, len(valid_classids)):
                result += number_classes[valid_classids[i]]
        else:
            first_valid_centerX = []
            first_valid_classids = []
            second_valid_centerX = []
            second_valid_classids = []
            for i in range(0, len(valid_centerX)):
                if (round(valid_centerY[i] + valid_centerH[i] / 2) < average_heightY):
                    first_valid_centerX.append(valid_centerX[i])
                    first_valid_classids.append(valid_classids[i])
                else:
                    second_valid_centerX.append(valid_centerX[i])
                    second_valid_classids.append(valid_classids[i])

            for i in range(0, len(first_valid_centerX)):
                for j in range(i + 1, len(first_valid_centerX)):
                    if first_valid_centerX[i] > first_valid_centerX[j]:
                        temp = first_valid_centerX[i]
                        first_valid_centerX[i] = first_valid_centerX[j]
                        first_valid_centerX[j] = temp

                        tem = first_valid_classids[i]
                        first_valid_classids[i] = first_valid_classids[j]
                        first_valid_classids[j] = tem
            for i in range(0, len(first_valid_classids)):
                result += number_classes[first_valid_classids[i]]

            for i in range(0, len(second_valid_centerX)):
                for j in range(i + 1, len(second_valid_centerX)):
                    if second_valid_centerX[i] > second_valid_centerX[j]:
                        temp = second_valid_centerX[i]
                        second_valid_centerX[i] = second_valid_centerX[j]
                        second_valid_centerX[j] = temp

                        tem = second_valid_classids[i]
                        second_valid_classids[i] = second_valid_classids[j]
                        second_valid_classids[j] = tem
            for i in range(0, len(second_valid_classids)):
                result += number_classes[second_valid_classids[i]]
        return result

    def save_license_to_db( license_str, server_id, camera_id):

        current_datetime = datetime.now()
        date_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

        if (temp_license_str != license_str):
            temp_license_str = license_str
    def check_min_value( a):
        if a < 0:
            return 0
        else:
            return a

    def check_X_max_value( x):
        if x > image_width:
            return image_width
        else:
            return x

    def check_Y_max_value( y):
        if y > image_height:
            return image_height
        else:
            return y

    def crop_image( img, x, y, w, h):
        x1 = check_min_value(x - int(w * 0.05))
        x2 = check_X_max_value(x + w + int(w * 0.05))
        y1 = check_min_value(y - int(h * 0.05))
        y2 = check_Y_max_value(y + h + int(h * 0.05))
        crop_img = img[y1:y2, x1:x2]
        return crop_img


    def process_plate_frame( image, car_type, origin_x, origin_y):
        scale = 0.00392
        retval, buffer = cv2.imencode('.png', image)
        base_64byte = base64.b64encode(buffer)
        base64_string = base_64byte.decode("utf-8")


        blob = cv2.dnn.blobFromImage(image, scale, (128, 128), (0, 0, 0), True, crop=False)
        plate_net.setInput(blob)
        outs = plate_net.forward(get_output_layers(plate_net))
        class_ids = []
        confidences = []
        boxes = []
        center_Y_list = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    center_Y_list.append(center_y)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        license_str = ""

        index = 0
        lpr_x = 0
        lpr_y = 0
        lpr_w = 0
        lpr_h = 0
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            color_plate = (0, 255, 0)

            plate_img = crop_image(image, round(x), round(y), round(w), round(h))
            tem_image = plate_img.copy()
            cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + h)), color_plate, 2)

            license_str = process_plate(plate_img, car_type)

        if len(license_str) > 7  and temp_result != license_str:

            #date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = f'{datetime.now():%d/%m/%Y}'
            file_path = date_string
            file_path.replace(":", "_")
            result = "LP_NO : " + license_str + '\n' + "Plate Type : " + car_type + '\n' +  "Date-Time : " + date_string

            cv2.imwrite("static/plate_result/" + license_str + ".jpg", tem_image)
            cv2.imwrite("static/vehicle_result/" + license_str + ".jpg", image)

            image = cv2.resize(image, (350, int(image.shape[0] * 350 / image.shape[1])))
            if len(image.shape) < 3 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, byteValue = image.shape
            byteValue = byteValue * width

            current_data = [license_str, car_type, date_string]
            data_info.insert(0, current_data)

            params = config()
            connection = psycopg2.connect(**params)
            cursor = connection.cursor()

            if get_param('save_db'):
                image_str = 'data:image/png;base64,' + base64_string
                sql = "INSERT INTO vehicle_data(vtype, number, image, date) VALUES ('" + car_type + "', '" + license_str + "', '" + image_str + "', '"  + date_string + "')"
                cursor.execute(sql)
                connection.commit()
                print('Saving into DB')

            if get_param('save_api'):
                payload = {'vtype': car_type, 'number': license_str, 'img': 'data:image/png;base64,' + base64_string, 'date': date_string}
                req = requests.post("http://test.eskoool.com/webservice.asmx/SaveANPR", data=payload)
                print(req.content)

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.HOST,self.PORT))
                    data = s.recv(1024)
                    data = data.decode('utf-8')
                    data = data.split(',')[4]
                    print('server data',data,type(data))
            except Exception as e:
                print(e)


            temp_result1 = license_str

    def process_frame( image):
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (128, 128), (0, 0, 0), True, crop=False)
        color_net.setInput(blob)
        outs = color_net.forward(get_output_layers(color_net))
        class_ids = []
        confidences = []
        boxes = []
        center_Y_list = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    center_Y_list.append(center_y)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            color_plate = (0, 255, 255)
            cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + h)), color_plate, 2)
            plate_color_img = crop_image(previousFrame, round(x), round(y), round(w), round(h))

            process_plate_frame(plate_color_img, color_classes[class_ids[i]], round(x), round(y))


    cap = cv2.VideoCapture(fname)
    i = 0
    framewidth = 1061
    while True:
        ret, frame = cap.read()
        if ret:
            i = i + 1
            previousFrame = frame.copy()
            image_width = frame.shape[1]
            image_height = frame.shape[0]
            frame_origin = frame
            
            if i % 3 == 0:
                process_frame(frame_origin)
                #print('Frame number,',i)
        else:
            break 
