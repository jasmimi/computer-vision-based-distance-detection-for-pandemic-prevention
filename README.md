# Computer vision-based distance detection for pandemic prevention
The 7 million death toll count of COVID-19 highlighted the importance of pandemic prevention measures. They already exist - masks, QR-code scanning, Bluetooth contact tracing, and manual tracing. Due to social and other factors, these effort-requiring existing methods become redunant. Various computer vision techniques can be integrated to create a system which processes video input to identify faces and calculate the distance between these faces. This solution requires minimal participation effort and is long lasting.

## How to run
Install modules
```
pip install -r requirements.txt
pip3 install opencv-python face_recognition
```

Run preferred file (uses webcam, no written file)
```
# standard speed
python integ_facerec_distance_faster.py

# faster by processing every other frame and scaling down input
python integ_facerec_distance_faster.py
```

## Credits
AUT ENSE891/ENSE892 Industrial Software Project 2024 with Professor Peter Chong

This project integrates and adapts code from the following Github Repositories:
- Face recognition https://github.com/ageitgey/face_recognition
- Distance detection https://github.com/Asadullah-Dal17/Distance_measurement_using_single_camera
- Object detection https://github.com/WongKinYiu/yolov7 
