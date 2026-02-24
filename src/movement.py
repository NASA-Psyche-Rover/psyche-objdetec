import RPi.GPIO as GPIO


class MotorController:
    """
    Minimal motor controller: forward / backward / stop.
    Uses your existing BCM pin mapping for left & right sides.
    """

    # LEFT side motors (BCM)
    L_IN1 = 23
    L_IN2 = 24
    L_IN3 = 5
    L_IN4 = 6

    # RIGHT side motors (BCM)
    R_IN1 = 12
    R_IN2 = 16
    R_IN3 = 20
    R_IN4 = 21

    ALL_PINS = [L_IN1, L_IN2, L_IN3, L_IN4, R_IN1, R_IN2, R_IN3, R_IN4]

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for p in self.ALL_PINS:
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)

        self._state = "stop"
        self.stop()

    def _left_forward(self):
        GPIO.output(self.L_IN1, GPIO.HIGH); GPIO.output(self.L_IN2, GPIO.LOW)
        GPIO.output(self.L_IN3, GPIO.HIGH); GPIO.output(self.L_IN4, GPIO.LOW)

    def _left_backward(self):
        GPIO.output(self.L_IN1, GPIO.LOW); GPIO.output(self.L_IN2, GPIO.HIGH)
        GPIO.output(self.L_IN3, GPIO.LOW); GPIO.output(self.L_IN4, GPIO.HIGH)

    def _right_forward(self):
        GPIO.output(self.R_IN1, GPIO.HIGH); GPIO.output(self.R_IN2, GPIO.LOW)
        GPIO.output(self.R_IN3, GPIO.HIGH); GPIO.output(self.R_IN4, GPIO.LOW)

    def _right_backward(self):
        GPIO.output(self.R_IN1, GPIO.LOW); GPIO.output(self.R_IN2, GPIO.HIGH)
        GPIO.output(self.R_IN3, GPIO.LOW); GPIO.output(self.R_IN4, GPIO.HIGH)

    def stop(self):
        if self._state == "stop":
            return
        for p in self.ALL_PINS:
            GPIO.output(p, GPIO.LOW)
        self._state = "stop"

    def forward(self):
        if self._state == "forward":
            return
        self._left_forward()
        self._right_forward()
        self._state = "forward"

    def backward(self):
        if self._state == "backward":
            return
        self._left_backward()
        self._right_backward()
        self._state = "backward"

    def cleanup(self):
        try:
            self.stop()
        finally:
            GPIO.cleanup()
