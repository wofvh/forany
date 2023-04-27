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

//상대값 위치만큼의 이동 	

int do_fas_MoveSingleAxisIncPos(int pulse) {
    unsigned char recv_data[255];
    ezi_move_t s;
    ezi_recv_t *r;
    int ret = -1;
    int rb = 0;
    int sb = 0;

    memset(&s,0x00,sizeof(ezi_move_t));

    s.stx = 0xAA;
    s.length =3+4+4;
    s.sync=ezi_sync++;
    s.reserved=0x00;
    s.frametype=FAS_MoveSingleAxisIncPos;

    s.speed = 10000;
    s.position = pulse;  //수정필요 

    printf("-----------> relative position = %d\n", s.position);
    sb = send(sockfd, &s, sizeof(s), 0);
    if (sb < 0) {
        printf("ERROR : send %s:%d [%s]\n", __func__, __LINE__, strerror(errno));
    } else {
        memset(&recv_data, 0x00, sizeof(recv_data));
        r = (ezi_recv_t *)recv_data;

        printf("OK : send %d byte\n", sb);
        rb = read(sockfd, recv_data, sizeof(recv_data));
        if (rb < 0) {
            printf("ERROR : recv[%d] %s:%d\n", rb, __func__, __LINE__);
        } else {
            printf("OK : recv[%d] %02x,%02x,%02x,%02x,%02x,%02x\n",
                   rb,
                   r->stx,
                   r->length,
                   r->sync,
                   r->reserved,
                   r->frametype,
                   r->status);
            switch (r->status) {
                case 0x00:
                    printf("cmd ok\n");
                    ret = 0;
                    break;
                case 0x80: printf("ERROR : wrong frametype\n"); break;
                case 0x81: printf("ERROR : range over date\n"); break;
                case 0x82: printf("ERROR : invalid frametype format\n"); break;
                case 0x85: printf("ERROR : drive faild\n"); break;
                case 0x86: printf("ERROR : reset failed\n"); break;
                case 0x87: printf("ERROR : servo on fail(1)\n"); break;
                case 0x88: printf("ERORR : servo on fail(2)\n"); break;
                case 0x89: printf("ERROR : servo on fail(3)\n"); break;
                default:
                    printf("ERROR : status = %02x\n", r->status);break;
            }
        }
    }
    return ret;
}
