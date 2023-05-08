#! /usr/bin/env python3.7

import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from std_msgs.msg import Int32MultiArray

class cmdfollow():
    def __init__(self):
        
        self.sub = rospy.Subscriber('/face_follow_cmd', String, self.callback, queue_size = 1)
        self.sub_drive = rospy.Subscriber('/control_ezi_drive_state', Bool, self.driveCallback, queue_size = 1)
        
        # self.pub_servo_jog = rospy.Publisher('/control_ezi_servo_jog_', Int32, queue_size = 1)
        self.pub_servo_jog = rospy.Publisher('/control_ezi_servo_jog_', Int32MultiArray, queue_size = 1)
        self.pub_servo_stop = rospy.Publisher('/control_ezi_servo_stop', Empty, queue_size = 1)
        self.pub_servo = rospy.Publisher('/control_ezi_servo', Int32, queue_size = 1)
        
        
        self.command_pre = '' # 
        self.cnt_repeat = 0
        self.speed = 10000
        
        self.drive = True
        
    def driveCallback(self, data):
        self.drive = data.data
        
    def callback(self, data):

        command = data.data
        
        # 여기서 바로 헤드부 컨트롤 쪽으로 메시지를 발행하는 방향으로 코드 작성
        if command != self.command_pre or not self.drive:
            if not self.drive:
                print('recieved drive-fail-msg, republish command')
            # 먼저 stop부터
            # rospy.Rate(20).sleep() # 여기는 필요 없을 지도 -> 바로바로 command가 바뀌지 않는 이상은
            if self.command_pre != 'stop' and self.drive: # drive fail로 다시 명령 시 stop은 빼도록 조건 추가
                msg = Empty()
                self.pub_servo_stop.publish(msg)
                print('publish stop')
            
            # stop 명령 후에 약간의 딜레이가 필요할지도 + -20~20 기준을 좀 더 넓힐 필요가 있을지도 -> 이건 테스트를 통해서 맞춰나가면 됨
            rospy.Rate(10).sleep()  # 여기는 필수, stop 이후에 딜레이가 있어야 메시지 발행이 안정적으로 됨.
            if command == 'right':
                # msg = Int32()
                # msg.data = 0
                msg = Int32MultiArray()
                msg.data = [0, self.speed]
                self.pub_servo_jog.publish(msg)
                print('publish turn right')
            elif command == 'left':
                # msg = Int32()
                # msg.data = 1
                msg = Int32MultiArray()
                msg.data = [1, self.speed]
                self.pub_servo_jog.publish(msg)
                print('publish turn left')
            elif command == 'left':
                # msg = Int32()
                # msg.data = 1
                msg = Int32MultiArray()
                msg.data = [1, 7000]
                self.pub_servo_jog.publish(msg)
                print('publish turn left')
            else:
                pass
                # msg = Empty()
                # self.pub_servo_stop.publish(msg)
                # print('publish stop(2)')
                
            self.command_pre = command
            self.drive = True
        else:
            # 동일한 상태 -> 아무것도 하지 않음, 여기에 딜레이가 들어가야 할지도..? 아닐수도.
            pass
            # 이 부분을 drive failed 시 false를 받아서 처리
            # if command == 'right' or command == 'left':
            #     self.cnt_repeat += 1
            #     if self.cnt_repeat > 100:
            #         self.command_pre = '' # self.command_pre를 초기화 함으로써 메시지를 발행하도록 함
            #         self.cnt_repeat = 0
            #         print('maybe error, reset command_pre')

        
    
def main():
    rospy.init_node('cmdfollow', anonymous=False)
    ic = cmdfollow()
    
    rospy.spin()
    
    
if __name__ == '__main__':
    main()
