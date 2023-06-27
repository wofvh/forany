#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_msgs.msg import Empty
from time import sleep

class GestureToSpeech():
    def __init__ (self):

        self.fingerLeftResult_ = ""
        self.fingerLeftResult_pre = ""
        self.noticetojeon_ = ""
        self.cnt_vision = 0
        self.cnt_left = 0
        # self.fingerRightReslut_ = ""
        self.grabtophoto = False
        
        self.isSaying = False
        self.time_map = {'Hi!': 7, 'Yeah!': 5, "Good":6, "Love":4, "Bad":6}
    
    def fingerLeftResultCallback(self, data):
        self.fingerLeftResult_ = data.data
        
        if self.fingerLeftResult_pre == self.fingerLeftResult_:
            self.cnt_left += 1
        else:
            self.cnt_left = 0
            
        self.fingerLeftResult_pre = self.fingerLeftResult_
        
    def grabtophotoCallback(self, data):
        self.grabtophoto =True
        
    def noticetojeon(self,data):
        self.noticetojeon_ = data.data
        if self.noticetojeon_ == "changed_to_vision_screen":
            # self.cnt_vision += 1
            self.isSaying = True
            # if self.cnt_vision >= 4:
            #     self.cnt_vision = 0
            #     if self.cnt_vision == 0:
            #         self.isSaying = False
                
            
            # rate =rospy.Rate(1.0/4.0)
            # rate.sleep()
        
def main():
    rospy.init_node('gesture_to_speech', anonymous=False)
    rateFloat = 5.0/1.0
    rate = rospy.Rate(rateFloat)
    ic = GestureToSpeech()
    
    # Pub
    noticeToJeonPublisher = rospy.Publisher("/notice_to_jeon", String, queue_size = 1)
    
    # Sub
    rospy.Subscriber("/notice_to_jeon", String,  ic.noticetojeon)
    rospy.Subscriber('/finger_left_result', String, ic.fingerLeftResultCallback)
    rospy.Subscriber('/grabbed_to_take_photo', Empty, ic.grabtophotoCallback)
    # rospy.Subscriber('/finger_right_result', String, ic.fingerRightResultCallback)

    # isSaying = False
    cnt_saying = 0
    cnt_photo = 0
    time_saying = 6

    while not rospy.is_shutdown():
            
        if ic.grabtophoto:
            cnt_photo += 1
            if cnt_photo >= 14 * rateFloat:
                ic.grabtophoto = False
                cnt_photo = 0
        else: 
            if not ic.isSaying:
                if ic.cnt_left >= 10:
                    if ic.fingerLeftResult_ in ['Hi!',"Yeah!","Good","Love","Bad"]:
                        noticeToJeonPublisher.publish(ic.fingerLeftResult_pre)
                        ic.cnt_left = 0
                        ic.isSaying = True
                        time_saying = ic.time_map[ic.fingerLeftResult_]
            
            else:
                cnt_saying += 1
                if cnt_saying >= time_saying * rateFloat:
                    ic.isSaying = False
                    cnt_saying = 0
            
            # if ic.grabtophoto:
            
            #     rospy.Rate(1/10).sleep()
                
            #     noticeToJeonPublisher.publish(ic.fingerLeftResult_pre)
            #     ic.grabtophoto = False
            #     noticeToJeonPublisher.publish(ic.fingerLeftResult_pre)
                       
                       
                  
        # if ic.can_publish
            
        

        rate.sleep()
        

if __name__ == '__main__':
    main()
    
    
