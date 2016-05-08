#include "standard.h"
int filter_size;
color *begin, *end, tmp[3], blkcpy2d_out[300];
#ifdef P6
#define INPUT(x,y) begin[((y)*(DIM*3))+(x)*3]
#endif

#ifdef P5
#define INPUT(x,y) begin[((y)*(DIM))+(x)]
#endif

#define OUTPUT(x,y) blkcpy2d_out[(y)*filter_size+(x)]
void load(int fd){
    int ct0a=4; struct stat _fstat;
    if (fstat(fd, &_fstat) == -1) { perror("fstat()"); exit(1); }
    color *p=mmap(NULL, _fstat.st_size, PROT_READ, MAP_PRIVATE, fd, 0), *a=p;
    if(!(a[0]=='P'&&a[1]==MagicNumber)) { fprintf(stderr, "Unsupported format\n"); exit(1); }
    do{ p=memchr(p, 0x0a, 64)+1; } while(--ct0a);
    fwrite(a,p-a,1,stdout);
    begin=p; end=a+_fstat.st_size;
}

int zero_mapping(int x, int y){
    return 0;
}

int buffered_mapping(int x, int y){
    if(x<0)x=0;
    else if(x>=DIM)x=(DIM-1);
    if(y<0)y=0;
    else if(y>=DIM)y=(DIM-1);
    return INPUT(x,y);
}

int wrapped_mapping(int x, int y){
    if(x<0)x+=DIM;
    else if(x>=DIM)x-=DIM;
    if(y<0)y+=DIM;
    else if(y>=DIM)y-=DIM-1;
    return INPUT(x,y);
}

void blkcpy2d(int _x, int _y){
    int r=(filter_size>>1),x1=_x-r,y1=_y-r,x2=_x+r,y2=_y+r;
    for(int y=y1;y<=y2;++y)for(int x=x1;x<=x2;++x)
        OUTPUT(x-x1,y-y1)=(x>=0&&x<DIM&&y>=0&&y<DIM)?INPUT(x,y):buffered_mapping(x,y);
}

#define ABS(x) ((x)<0?-(x):(x))
#define fScale (1.f)
int sobel_filter(int x, int y){
    blkcpy2d(x, y);
    short Horz = OUTPUT(2,0) + 2*OUTPUT(2,1) + OUTPUT(2,2) - OUTPUT(0,0) - 2*OUTPUT(0,1) - OUTPUT(0,2);
    short Vert = OUTPUT(0,0) + 2*OUTPUT(1,0) + OUTPUT(2,0) - OUTPUT(0,2) - 2*OUTPUT(1,2) - OUTPUT(2,2);
    short Sum = (short) (fScale*(ABS(Horz)+ABS(Vert)));
    if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
    return (int) (unsigned char) Sum;
}

int main(int argc, char *argv[]){
    if(argc<2){ fprintf(stderr, "Usage: input.ppm\n"); return 1; }
    filter_size=3; load(open(argv[1], O_RDONLY));
    for(int y=0;y<DIM;++y)for(int x=0;x<DIM;++x){
#ifdef P6
        memset(tmp, sobel_filter(x,y), 3);
        fwrite(tmp, 3, 1, stdout);
#endif

#ifdef P5
        memset(tmp, sobel_filter(x,y), 1);
        fwrite(tmp, 1, 1, stdout);
#endif
    }
    return 0;
}
