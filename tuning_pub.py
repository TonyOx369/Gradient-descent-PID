#!/usr/bin/python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from pid_controller import PIDController
from gradient_descent import GradientDescent
import math
from std_srvs.srv import Empty
from std_msgs.msg import Float64
import matplotlib.pyplot as plt

class TurtleCostFunction:
    def __init__(self):
        self.a = np.array([0.75, 0, 0])
        self.target_x = 7.5
        self.dt = 0.05
        self.pid_controller = PIDController()
        self.current_x = None
        self.cost = 0.0
        self.cost_h = 0
        self.gradient_descent = GradientDescent(0.001)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)
        self.reset_service = rospy.ServiceProxy("/reset", Empty)
        self.process_variables = []
        self.iter = 0
        self.cost_vals = []
        self.cost_pub = rospy.Publisher('/turtle_cost', Float64, queue_size=10)
        self.kp_pub = rospy.Publisher('/kp', Float64, queue_size=10)
        self.ki_pub = rospy.Publisher('/ki', Float64, queue_size=10)
        self.kd_pub = rospy.Publisher('/kd', Float64, queue_size=10)

    def pose_callback(self, data):
        self.current_x = data.x
        self.process_variables.append(self.current_x)

    def execute(self):
        rate = rospy.Rate(20)
        n = 0
        _h = True
        while not rospy.is_shutdown():
            velocity_msg = Twist()
            if self.current_x is not None:
                if _h:
                    if n < 250:
                        velocity_msg.linear.x = self.pid_controller.compute_control(self.target_x, self.current_x, self.dt, self.a)
                        self.velocity_publisher.publish(velocity_msg)
                        self.cost += math.sqrt(pow((self.target_x - self.current_x) * self.dt, 2))
                        n+=1
                    else:
                        n = 0
                        _h = False
                        self.reset_service.call()
                        h = 0.0000001
                        a_h = [0, 0, 0]
                        a_h[0] = self.a[0] + h
                        a_h[1] = self.a[1] + h
                        a_h[2] = self.a[2] + h      
                        print("----finding _h cost----")
                else:
                    if n < 250:
                        velocity_msg.linear.x = self.pid_controller.compute_control(self.target_x, self.current_x, self.dt, a_h)
                        self.velocity_publisher.publish(velocity_msg)
                        self.cost_h += (self.target_x - self.current_x) * self.dt
                        n+=1
                    else:
                        self.a = self.gradient_descent.execute_adagrad(self.a, self.cost, self.cost_h)
                        n = 0 
                        self.iter+=1
                        self.cost_vals.append(self.cost)
                        self.cost_pub.publish(self.cost)  # Publish cost info
                        self.kp_pub.publish(self.a[0])
                        self.ki_pub.publish(self.a[2])
                        self.kd_pub.publish(self.a[1])
                        _h = True
                        self.reset_service.call()
                        self.cost = 0
                        self.cost_h = 0
                        print("----finding normal cost----")

                print("n: " + str(n))
                print("iteration: "+str(self.iter))
                print("kp: " + str(self.a[0]) + " kd: " + str(self.a[1]) + " ki: " + str(self.a[2]))
                print("cost: " + str(self.cost))
                print("G: " + str(self.gradient_descent.G))
                print("learning_rate: " + str(self.gradient_descent.learning_rate))
                self.update_plot()

            rate.sleep()

    def update_plot(self):
        plt.clf()
        plt.plot([0, len(self.process_variables) - 1], [self.target_x, self.target_x], 'r--', label='Setpoint')  # Plotting the setpoint as a constant line
        plt.plot(self.process_variables, 'b-', label='Process Variable')  # Plotting the process variable curve
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Set Variable vs. Process Variable')
        plt.grid(True)
        plt.pause(0.01)
        plt.legend()

if __name__ == "__main__":
    rospy.init_node("cool")
    pid_tune = TurtleCostFunction()
    pid_tune.execute()
