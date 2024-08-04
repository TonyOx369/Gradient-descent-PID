class PIDController:
    def __init__(self):

        self.prev_error = 0
        self.integral = 0

    def compute_control(self, setpoint, process_variable, dt, a):
        self.Kp = a[0]
        # self.Ki = a[1]
        # self.Kd = a[2]
        self.Ki = 0
        self.Kd = 0
        error = setpoint - process_variable

        # Proportional Term
        P = self.Kp * error

        # Integral Term
        self.integral += error * dt
        I = self.Ki * self.integral

        # Derivative Term
        D = self.Kd * (error - self.prev_error)/dt
        self.prev_error = error

        control = P + I + D
        return float(control)


