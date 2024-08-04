#!/usr/bin/python3

import rospy
import matplotlib as m
m.use('Qt5Agg')
import matplotlib.pyplot as plt

from std_msgs.msg import Float64

class CostPlotter:
    def __init__(self):
        self.cost_vals = []
        self.kp_vals = []
        self.kd_vals = []
        self.ki_vals = []

        rospy.init_node("cost_subscriber")
        rospy.Subscriber('/kp', Float64, self.kp_callback)
        rospy.Subscriber('/kd', Float64, self.kd_callback)
        rospy.Subscriber('/ki', Float64, self.ki_callback)
        rospy.Subscriber('/turtle_cost', Float64, self.cost_callback)

    def cost_callback(self, cost_msg):
        self.cost_vals.append(cost_msg.data)
        self.plot_graph()

    def kp_callback(self, kp_msg):
        self.kp_vals.append(kp_msg.data)

    def kd_callback(self, kd_msg):
        self.kd_vals.append(kd_msg.data)

    def ki_callback(self, ki_msg):
        self.ki_vals.append(ki_msg.data)

    def plot_graph(self):
        plt.plot(range(len(self.cost_vals)), self.cost_vals, marker='o', label='Cost')
        plt.title('Iteration vs. Cost')
        plt.clf()
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)  # Pause to allow plot to update
        # plt.ion()
        plt.draw()

if __name__ == "__main__":
    cost_plotter = CostPlotter()
    rospy.spin()
