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
        
        self.cnt_left = 0
        # self.fingerRightReslut_ = ""
        self.grabtophoto = False
        
    
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
            rospy.Rate(1.0/4.0).sleep()
        
    
    # def fingerLeftResultCallback(self, data):
    #     self.fingerLeftResult_pre = data.data
        
        # if self.fingerLeftResult_pre is not None:
            
        #     self.fingerLeftResult_ = self.fingerLeftResult_(self.fingerLeftResult_per, data)
           


    # def fingerRightResultCallback(self, data):
    #     self.fingerRightResult_ = data.data

        
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

    isSaying = False
    cnt_saying = 0
    cnt_photo = 0
    cnt_vision = 0

    while not rospy.is_shutdown():
            
        if ic.grabtophoto:
            cnt_photo += 1
            if cnt_photo >= 10 * rateFloat:
                ic.grabtophoto = False
                cnt_photo = 0
        else: 
            if not isSaying:
                if ic.cnt_left >= 10:
                    if ic.fingerLeftResult_ in ['Hi!',"Yeah!","Good","Love","Bad"]:
                        noticeToJeonPublisher.publish(ic.fingerLeftResult_pre)
                        ic.cnt_left = 0
                        isSaying = True
            
                else:
                    cnt_saying += 1
                    if cnt_saying >= 10 * rateFloat:
                        isSaying = False
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
    
    
