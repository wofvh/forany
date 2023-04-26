// https://docs.microsoft.com/ko-kr/dotnet/framework/network-programming/asynchronous-server-socket-example
#include <ros/ros.h>
#include <ros/console.h>
#include <iostream>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <netinet/in.h>
#include <arpa/inet.h>


#include "ezi_packet.h"
#include "ezi_frametype.h"

#include <std_msgs/Int32.h>
#include <std_msgs/Empty.h>

extern int errno;
#define MAX 80
#define PORT 2002
#define SA struct sockaddr

int sockfd;
int ezi_sync=1;
int do_fas_getboardinfo()
{
    unsigned char recv_data[255];
    ezi_send_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    memset(&s,0x00,sizeof(ezi_send_t));
    memset(&recv_data,0x00,sizeof(recv_data));
    r=(ezi_recv_t *)recv_data;
    s.stx=0xAA;
    s.length=3;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_GetboardInfo;

    sb=send(sockfd,&s,5,0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
        printf("OK : send %d byte\n",sb);
        rb=read(sockfd,recv_data,sizeof(recv_data));
        if (rb<0)
        {
            printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
        }
        else {
            printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
                    rb,
                    r->stx,
                    r->length,
                    r->sync,
                    r->reserved,
                    r->frametype,
                    r->status);
            switch (r->status)
            {
                case 0x00: printf("cmd ok\n"); ret=0; break;
                case 0x80: printf("ERROR : wrong frametype\n"); break;
                case 0x81: printf("ERROR : range over date\n"); break;
                case 0x82: printf("ERROR : invalid frametype format\n"); break;
                case 0x85: printf("ERROR : drive faild\n"); break;
                case 0x86: printf("ERROR : reset failed\n"); break;
                case 0x87: printf("ERROR : servo on fail(1)\n"); break;
                case 0x88: printf("ERORR : servo on fail(2)\n"); break;
                case 0x89: printf("ERROR : servo on fail(3)\n"); break;
                default: printf("ERROR : status = %02x\n",r->status); break;
            }
        }
    }
    return ret;
}

int do_fas_getax()
{
    unsigned char recv_data[255];
    ezi_send_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    int i;

    int axisstatus=0;
    memset(&s,0x00,sizeof(ezi_send_t));
    memset(&recv_data,0x00,sizeof(recv_data));
    r=(ezi_recv_t *)recv_data;
    s.stx=0xAA;
    s.length=3;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_GetAxisStatus;

    sb=send(sockfd,&s,5,0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
        printf("OK : send %d byte\n",sb);
        rb=read(sockfd,recv_data,sizeof(recv_data));
        if (rb<0)
        {
            printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
        }
        else {
            printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
                    rb,
                    r->stx,
                    r->length,
                    r->sync,
                    r->reserved,
                    r->frametype,
                    r->status);
            switch (r->status)
            {
                case 0x00: printf("cmd ok\n"); ret=0; break;
                case 0x80: printf("ERROR : wrong frametype\n"); break;
                case 0x81: printf("ERROR : range over date\n"); break;
                case 0x82: printf("ERROR : invalid frametype format\n"); break;
                case 0x85: printf("ERROR : drive faild\n"); break;
                case 0x86: printf("ERROR : reset failed\n"); break;
                case 0x87: printf("ERROR : servo on fail(1)\n"); break;
                case 0x88: printf("ERORR : servo on fail(2)\n"); break;
                case 0x89: printf("ERROR : servo on fail(3)\n"); break;
                default: printf("ERROR : status = %02x\n",r->status); break;
            }
        axisstatus=recv_data[sizeof(ezi_recv_t)];

	for (i=0;i<rb;i++)
	{
		printf("%02x\n",recv_data[i]);
	}

        printf("axisstatus = %08x: %s:%d\n",axisstatus,__func__,__LINE__);
        if (axisstatus&FFLAG_ERRORALL)
        {
                printf("ERROR : FFLAG_ERRORALL %s:%d\n",__func__,__LINE__);
        }
        if(axisstatus&FFLAG_SERVOON)
        {
                printf("FFLAG_SERVOON(enable) = %08x: %s:%d\n",axisstatus,__func__,__LINE__);
        }
        else
                printf("FFLAG_SERVOON(disable) = %08x: %s:%d\n",axisstatus,__func__,__LINE__);

        }
    }
    return ret;
}

int do_fas_servoalarmreset()
{
    unsigned char recv_data[255];
    ezi_send_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    int i;
    int axisstatus=0;
    memset(&s,0x00,sizeof(ezi_send_t));
    memset(&recv_data,0x00,sizeof(recv_data));
    r=(ezi_recv_t *)recv_data;
    s.stx=0xAA;
    s.length=3;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_ServoAlarmReset;

    sb=send(sockfd,&s,sizeof(s),0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
	    printf("OK : send %d byte\n",sb);
	    rb=read(sockfd,recv_data,sizeof(recv_data));
	    if (rb<0)
	    {
		    printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
	    }
	    else {
		    printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
				    rb,
				    r->stx,
				    r->length,
				    r->sync,
				    r->reserved,
				    r->frametype,
				    r->status);
		    switch (r->status)
		    {
			    case 0x00: printf("cmd ok\n"); ret=0; break;
			    case 0x80: printf("ERROR : wrong frametype\n"); break;
			    case 0x81: printf("ERROR : range over date\n"); break;
			    case 0x82: printf("ERROR : invalid frametype format\n"); break;
			    case 0x85: printf("ERROR : drive faild\n"); break;
			    case 0x86: printf("ERROR : reset failed\n"); break;
			    case 0x87: printf("ERROR : servo on fail(1)\n"); break;
			    case 0x88: printf("ERORR : servo on fail(2)\n"); break;
			    case 0x89: printf("ERROR : servo on fail(3)\n"); break;
			    default: printf("ERROR : status = %02x\n",r->status); break;
		    }
		    for (i=0;i<rb;i++)
		    {
			    printf("%02x\n",recv_data[i]);
		    }
	    }
    }
    return ret;
}


int do_fas_servoenable()
{
    unsigned char recv_data[255];
    ezi_servo_enable_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    int i;
    int axisstatus=0;
    memset(&s,0x00,sizeof(ezi_servo_enable_t));
    memset(&recv_data,0x00,sizeof(recv_data));
    r=(ezi_recv_t *)recv_data;
    s.stx=0xAA;
    s.length=4;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_ServoEnable;
    s.enable=1;

    sb=send(sockfd,&s,sizeof(s),0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
	    printf("OK : send %d byte\n",sb);
	    rb=read(sockfd,recv_data,sizeof(recv_data));
	    if (rb<0)
	    {
		    printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
	    }
	    else {
		    printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
				    rb,
				    r->stx,
				    r->length,
				    r->sync,
				    r->reserved,
				    r->frametype,
				    r->status);
		    switch (r->status)
		    {
			    case 0x00: printf("cmd ok\n"); ret=0; break;
			    case 0x80: printf("ERROR : wrong frametype\n"); break;
			    case 0x81: printf("ERROR : range over date\n"); break;
			    case 0x82: printf("ERROR : invalid frametype format\n"); break;
			    case 0x85: printf("ERROR : drive faild\n"); break;
			    case 0x86: printf("ERROR : reset failed\n"); break;
			    case 0x87: printf("ERROR : servo on fail(1)\n"); break;
			    case 0x88: printf("ERORR : servo on fail(2)\n"); break;
			    case 0x89: printf("ERROR : servo on fail(3)\n"); break;
			    default: printf("ERROR : status = %02x\n",r->status); break;
		    }
		    for (i=0;i<rb;i++)
		    {
			    printf("%02x\n",recv_data[i]);
		    }
	    }
    }
    return ret;
}

int do_fas_movesingle(int degree)
{
    unsigned char recv_data[255];
    ezi_move_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    int axisstatus=0;

    memset(&s,0x00,sizeof(ezi_move_t));

    s.stx=0xAA;
    s.length=3+4+4;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_MoveSingleAxisAbsPos;

    s.speed=10000;
    //int max=815500;
    //int max=655350;
    int max=140000;
    s.position=(max*(degree-27))/360;
    
//    s.position=407750;
    printf("-----------> position = %d\n",s.position);
    sb=send(sockfd,&s,sizeof(s),0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
    	memset(&recv_data,0x00,sizeof(recv_data));
    	r=(ezi_recv_t *)recv_data;

        printf("OK : send %d byte\n",sb);
        rb=read(sockfd,recv_data,sizeof(recv_data));
        if (rb<0)
        {
            printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
        }
        else {
            printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
                    rb,
                    r->stx,
                    r->length,
                    r->sync,
                    r->reserved,
                    r->frametype,
                    r->status);
            switch (r->status)
            {
                case 0x00: printf("cmd ok\n"); ret=0; break;
                case 0x80: printf("ERROR : wrong frametype\n"); break;
                case 0x81: printf("ERROR : range over date\n"); break;
                case 0x82: printf("ERROR : invalid frametype format\n"); break;
                case 0x85: printf("ERROR : drive faild\n"); break;
                case 0x86: printf("ERROR : reset failed\n"); break;
                case 0x87: printf("ERROR : servo on fail(1)\n"); break;
                case 0x88: printf("ERORR : servo on fail(2)\n"); break;
                case 0x89: printf("ERROR : servo on fail(3)\n"); break;
                default: printf("ERROR : status = %02x\n",r->status); break;
            }
        }
    }
    return ret;
}

int do_fas_movevelocity(char direction)
{
    unsigned char recv_data[255];
    ezi_movevelocity_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    int axisstatus=0;

    memset(&s,0x00,sizeof(ezi_move_t));

    s.stx=0xAA;
    s.length=3+4+1;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_MoveVelocity;

    s.speed=10000;
    s.direction=direction;
    
//    s.position=407750;
    printf("-----------> direction = %d , speed = %d \n", s.direction, s.speed);
    sb=send(sockfd,&s,sizeof(s),0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
    	memset(&recv_data,0x00,sizeof(recv_data));
    	r=(ezi_recv_t *)recv_data;

        printf("OK : send %d byte\n",sb);
        rb=read(sockfd,recv_data,sizeof(recv_data));
        if (rb<0)
        {
            printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
        }
        else {
            printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
                    rb,
                    r->stx,
                    r->length,
                    r->sync,
                    r->reserved,
                    r->frametype,
                    r->status);
            switch (r->status)
            {
                case 0x00: printf("cmd ok\n"); ret=0; break;
                case 0x80: printf("ERROR : wrong frametype\n"); break;
                case 0x81: printf("ERROR : range over date\n"); break;
                case 0x82: printf("ERROR : invalid frametype format\n"); break;
                case 0x85: printf("ERROR : drive faild\n"); break;
                case 0x86: printf("ERROR : reset failed\n"); break;
                case 0x87: printf("ERROR : servo on fail(1)\n"); break;
                case 0x88: printf("ERORR : servo on fail(2)\n"); break;
                case 0x89: printf("ERROR : servo on fail(3)\n"); break;
                default: printf("ERROR : status = %02x\n",r->status); break;
            }
        }
    }
    return ret;
}

int do_fas_movestop(){
    unsigned char recv_data[255];
    ezi_movestop_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    int axisstatus=0;

    memset(&s,0x00,sizeof(ezi_move_t));

    s.stx=0xAA;
    s.length=3;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_MoveStop;

    
//    s.position=407750;
    printf("-----------> stop \n");
    sb=send(sockfd,&s,sizeof(s),0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
    	memset(&recv_data,0x00,sizeof(recv_data));
    	r=(ezi_recv_t *)recv_data;

        printf("OK : send %d byte\n",sb);
        rb=read(sockfd,recv_data,sizeof(recv_data));
        if (rb<0)
        {
            printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
        }
        else {
            printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
                    rb,
                    r->stx,
                    r->length,
                    r->sync,
                    r->reserved,
                    r->frametype,
                    r->status);
            switch (r->status)
            {
                case 0x00: printf("cmd ok\n"); ret=0; break;
                case 0x80: printf("ERROR : wrong frametype\n"); break;
                case 0x81: printf("ERROR : range over date\n"); break;
                case 0x82: printf("ERROR : invalid frametype format\n"); break;
                case 0x85: printf("ERROR : drive faild\n"); break;
                case 0x86: printf("ERROR : reset failed\n"); break;
                case 0x87: printf("ERROR : servo on fail(1)\n"); break;
                case 0x88: printf("ERORR : servo on fail(2)\n"); break;
                case 0x89: printf("ERROR : servo on fail(3)\n"); break;
                default: printf("ERROR : status = %02x\n",r->status); break;
            }
        }
    }
    return ret;
}

int do_fas_originsingle()
{
    unsigned char recv_data[255];
    ezi_send_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    int axisstatus=0;
    memset(&s,0x00,sizeof(ezi_send_t));
    memset(&recv_data,0x00,sizeof(recv_data));
    r=(ezi_recv_t *)recv_data;
    s.stx=0xAA;
    s.length=3;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_MoveOriginSingleAxis;

    sb=send(sockfd,&s,sizeof(s),0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
        printf("OK : send %d byte\n",sb);
        rb=read(sockfd,recv_data,sizeof(recv_data));
        if (rb<0)
        {
            printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
        }
        else {
            printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
                    rb,
                    r->stx,
                    r->length,
                    r->sync,
                    r->reserved,
                    r->frametype,
                    r->status);
            switch (r->status)
            {
                case 0x00: printf("cmd ok\n"); ret=0; break;
                case 0x80: printf("ERROR : wrong frametype\n"); break;
                case 0x81: printf("ERROR : range over date\n"); break;
                case 0x82: printf("ERROR : invalid frametype format\n"); break;
                case 0x85: printf("ERROR : drive faild\n"); break;
                case 0x86: printf("ERROR : reset failed\n"); break;
                case 0x87: printf("ERROR : servo on fail(1)\n"); break;
                case 0x88: printf("ERORR : servo on fail(2)\n"); break;
                case 0x89: printf("ERROR : servo on fail(3)\n"); break;
                default: printf("ERROR : status = %02x\n",r->status); break;
            }
	    }
    }
    return ret;
}

int do_fas_movetolimit()
{
    unsigned char recv_data[255];
    ezi_move2limit_t s;
    ezi_recv_t *r;
    int ret=-1;
    int rb=0;
    int sb=0;

    int axisstatus=0;
    memset(&s,0x00,sizeof(ezi_move2limit_t));
    memset(&recv_data,0x00,sizeof(recv_data));
    r=(ezi_recv_t *)recv_data;
    s.stx=0xAA;
    s.length=3+4+1;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_MoveToLimit;
    s.speed=0x0000ffff;
    s.limit=1;

    sb=send(sockfd,&s,sizeof(s),0);
    if (sb<0)
    {
        printf("ERROR : send %s:%d [%s]\n",__func__,__LINE__,strerror(errno));
    }
    else
    {
        printf("OK : send %d byte\n",sb);
        rb=read(sockfd,recv_data,sizeof(recv_data));
        if (rb<0)
        {
            printf("ERROR : recv[%d] %s:%d\n",rb,__func__,__LINE__);
        }
        else {
            printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
                    rb,
                    r->stx,
                    r->length,
                    r->sync,
                    r->reserved,
                    r->frametype,
                    r->status);
            switch (r->status)
            {
                case 0x00: printf("cmd ok\n"); ret=0; break;
                case 0x80: printf("ERROR : wrong frametype\n"); break;
                case 0x81: printf("ERROR : range over date\n"); break;
                case 0x82: printf("ERROR : invalid frametype format\n"); break;
                case 0x85: printf("ERROR : drive faild\n"); break;
                case 0x86: printf("ERROR : reset failed\n"); break;
                case 0x87: printf("ERROR : servo on fail(1)\n"); break;
                case 0x88: printf("ERORR : servo on fail(2)\n"); break;
                case 0x89: printf("ERROR : servo on fail(3)\n"); break;
                default: printf("ERROR : status = %02x\n",r->status); break;
            }
        // axisstatus = ( ezi_recv_t * )( recv + 1 );
        // printf("axisstatus = %08x: %s:%d\n",axisstatus,__func__,__LINE__);
        // if (axisstatus&FFLAG_ERRORALL)
        // {
		// 	printf("ERROR : FFLAG_ERRORALL %s:%d\n",__func__,__LINE__);
        // }
        // if(axisstatus&FFLAG_SERVOON)
        // {
		// 	printf("FFLAG_SERVOON(enable) = %08x: %s:%d\n",axisstatus,__func__,__LINE__);
        // }
        // else
		// 	printf("FFLAG_SERVOON(disable) = %08x: %s:%d\n",axisstatus,__func__,__LINE__);

        }
    }
    return ret;
}

void controlEziServoCallback(const std_msgs::Int32 & msg){
	
	if (msg.data == -1){
		printf("origin single\n");
	    do_fas_originsingle();
	}
	else {
		std::cout << "move single to " << msg.data << std::endl;
		do_fas_movesingle(msg.data);
	}
}

void controlEziServoJogCallback(const std_msgs::Int32 & msg){
    do_fas_movevelocity(msg.data);
}

void controlEziServoStopCallback(const std_msgs::Empty & msg){
    do_fas_movestop();
}


int main (int argc, char** argv)
{
	ros::init(argc, argv, "ezi_servo_controller");
    ros::NodeHandle nh;
    ros::Rate loop_rate(10);

	ros::Subscriber controlEziServoSubscriber = nh.subscribe("/control_ezi_servo", 10, controlEziServoCallback);
    ros::Subscriber controlEziServoJogSubscriber = nh.subscribe("/control_ezi_servo_jog", 10, controlEziServoJogCallback);
    ros::Subscriber controlEziServoStopSubscriber = nh.subscribe("/control_ezi_servo_stop", 10, controlEziServoStopCallback);
	
	struct sockaddr_in servaddr, cli;

    // socket create and verification
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");
    bzero(&servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("192.168.10.9");
    servaddr.sin_port = htons(PORT);

    // connect the client socket to server socket
    if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) {
        printf("connection with the server failed...\n");
        exit(0);
    }
    else
        printf("connect OK\n");

    //do_fas_getboardinfo();
    sleep(1);
    do_fas_getax();
    printf("servoalarmreset\n");
    do_fas_servoalarmreset();

    printf("servoenable\n");
    do_fas_servoenable();
    sleep(1);
    
    do_fas_originsingle();
    sleep(1);
    //do_fas_movesingle(-20);
	
	while(ros::ok())
    {
		ros::spinOnce();

        loop_rate.sleep();
    }

	close(sockfd);
}

	
