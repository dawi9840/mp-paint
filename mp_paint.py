import mediapipe as mp
import cv2
import numpy as np
import time
import math
from google.protobuf.json_format import MessageToDict


def camera_info(cap):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'w: {w}, h: {h}')

    while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Raw Video Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    print('Done.')
    cap.release()
    cv2.destroyAllWindows()


def mp_holistic_detections(cap):
    ''' Mediapipe holistic detections with a video (or camera). '''
    # Color difine
    color_pose1 = (245,117,66)
    color_pose2 = (245,66,230)

    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # landmarks = results.pose_landmarks.landmark
                # print(f'nose_x: {landmarks[0].x}')
                # print(f'nose_y: {landmarks[0].y}')

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_pose1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_pose2, thickness=2, circle_radius=2)
                )

                cv2.imshow('Raw Video Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
            
    print('Done.')
    cap.release()
    cv2.destroyAllWindows()


def mp_hand_detections(cap):
    ''' Mediapipe hands detections with a video (or camera). '''
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')
    
    # https://blog.csdn.net/weixin_45930948/article/details/115444916
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
    
    while True:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.flip(frame,1)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_handedness:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                print(type(handedness_dict))
                print(f'idx: {idx}')
                print(handedness_dict['classification'][0])
                print(handedness_dict['classification'][0]['score'])
            # for hand_label in results.multi_handedness:
            #     # print(f'type: {type(hand_label)}')
            #     print(f'hand_label:\n {hand_label}')
            #     # print('hand_label: ', hand_label['index'])

        # hand_landmarks visualization
        frame.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # print(f'hand visualization:\n {hand_landmarks}')

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    print('Done!')
    cap.release()
    cv2.destroyAllWindows()


class ColorNumbers:
	''' RGB color numbers. '''
	light_blue = (50,152,255)
	cyan = (0,255,255)
	yellow = (255,255,0)
	green = (0,255,0)
	blue = (0,0,255)
	black = (0,0,0)


def getTool(x, ml):
	''' Get tools function. '''
	if x < 50 + ml:
		return "line"

	elif x<100 + ml:
		return "rectangle"

	elif x < 150 + ml:
		return"draw"

	elif x<200 + ml:
		return "circle"

	else:
		return "erase"


def index_raised(yi, y9):
	if (y9 - yi) > 40:
		return True

	return False


def virtual_paint(cap):
	# contants
	ml = 150
	max_x, max_y = 250+ml, 50
	curr_tool = "select tool"
	time_init = True
	rad = 40
	var_inits = False
	thick = 4
	prevx, prevy = 0,0

	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	mp_hands = mp.solutions.hands
	hand_landmark = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
	mp_draw = mp.solutions.drawing_utils


	# drawing tools
	tools = cv2.imread("tools.png")
	tools = tools.astype('uint8') # converted to unsigned int 8

	mask = np.ones((h, w)) * 255
	mask = mask.astype('uint8')

	color = ColorNumbers()
	'''
	tools = np.zeros((max_y+5, max_x+5, 3), dtype="uint8")
	cv2.rectangle(tools, (0,0), (max_x, max_y), color.blue, 2)
	cv2.line(tools, (50,0), (50,50), color.blue, 2)
	cv2.line(tools, (100,0), (100,50), color.blue, 2)
	cv2.line(tools, (150,0), (150,50), color.blue, 2)
	cv2.line(tools, (200,0), (200,50), color.blue, 2)
	'''

	while True:
		_, frm = cap.read()
		frm = cv2.flip(frm, 1)

		rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

		op = hand_landmark.process(rgb)

		if op.multi_hand_landmarks:
			for i in op.multi_hand_landmarks:
				mp_draw.draw_landmarks(frm, i, mp_hands.HAND_CONNECTIONS)
				x, y = int(i.landmark[8].x*w), int(i.landmark[8].y*h)

				if x < max_x and y < max_y and x > ml:
					if time_init:
						ctime = time.time()
						time_init = False
					ptime = time.time()

					cv2.circle(frm, (x, y), rad, color.cyan, 2)
					rad -= 1

					if (ptime - ctime) > 0.8:
						curr_tool = getTool(x, ml)
						print("your current tool set to : ", curr_tool)
						time_init = True
						rad = 40

				else:
					time_init = True
					rad = 40

				if curr_tool == "draw":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
						prevx, prevy = x, y

					else:
						prevx = x
						prevy = y

				elif curr_tool == "line":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						if not(var_inits):
							xii, yii = x, y
							var_inits = True

						cv2.line(frm, (xii, yii), (x, y), color.light_blue, thick)

					else:
						if var_inits:
							cv2.line(mask, (xii, yii), (x, y), 0, thick)
							var_inits = False

				elif curr_tool == "rectangle":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						if not(var_inits):
							xii, yii = x, y
							var_inits = True

						cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)

					else:
						if var_inits:
							cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
							var_inits = False

				elif curr_tool == "circle":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						if not(var_inits):
							xii, yii = x, y
							var_inits = True

						cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), color.yellow, thick)

					else:
						if var_inits:
							cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), color.green, thick)
							var_inits = False

				elif curr_tool == "erase":
					xi, yi = int(i.landmark[12].x*w), int(i.landmark[12].y*h)
					y9  = int(i.landmark[9].y*h)

					if index_raised(yi, y9):
						# cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
						cv2.circle(frm, (x, y), 30, color.black, -1)
						cv2.circle(mask, (x, y), 30, 255, -1)

		# https://www.cxyzjd.com/article/zhouzongzong/93028651
		# https://medium.com/linux-on-raspberry-pi4/opencv-bitwise-and-%E7%9A%84mask%E9%80%9A%E4%BF%97%E7%90%86%E8%A7%A3-%E8%BD%89%E9%8C%84-d19cd63f34f0
		# https://ithelp.ithome.com.tw/articles/10246072
		op = cv2.bitwise_and(frm, frm, mask=mask)
		frm[:, :, 1] = op[:, :, 1]
		frm[:, :, 2] = op[:, :, 2]

		frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

		cv2.putText(frm, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color.blue, 2)
		cv2.imshow("paint app", frm)

		if cv2.waitKey(1) & 0XFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
    # camera_info(cap=cv2.VideoCapture(0))
    # mp_holistic_detections(cap=cv2.VideoCapture(0))
    mp_hand_detections(cap=cv2.VideoCapture(0))

    
   