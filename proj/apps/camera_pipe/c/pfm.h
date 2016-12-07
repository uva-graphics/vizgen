
/* Read/write 1 and 3 channel PFM files, public domain Connelly Barnes 2007. */

#ifndef _pfm_h
#define _pfm_h


/* Read/write 1 and 3 channel PFM files, public domain Connelly Barnes 2007. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef unsigned char byte;

int is_little_endian() {
  if (sizeof(float) != 4) { printf("Bad float size.\n"); exit(1); }
  byte b[4] = { 255, 0, 0, 0 };
  return *((float *) b) < 1.0;
}

float *read_pfm_file(const char *filename, int *w, int *h) {
  char buf[256];
  int i, ans;
  FILE *f = fopen(filename, "rb");
  ans = fscanf(f, "%s\n", buf);
  if (strcmp(buf, "Pf") != 0) {
    //printf("Not a 1 channel PFM file.\n");
    return NULL;
  }
  ans = fscanf(f, "%d %d\n", w, h);
  double scale = 1.0;
  ans = fscanf(f, "%lf\n", &scale);
  int little_endian = 0;
  if (scale < 0.0) {
    little_endian = 1;
    scale = -scale;
  }
  byte *data;
  data = malloc((*w) * (*h) * 4 * sizeof(byte));
  float *depth;
  depth = malloc((*w) * (*h) * sizeof(float));
  int count = fread((void *) data, 4, (*w)*(*h), f);
  if (count != (*w)*(*h)) {
    printf("Error reading PFM file.\n"); return NULL;
  }
  int native_little_endian = is_little_endian();
  for (i = 0; i < (*w)*(*h); i++) {
    byte *p = &data[i*4];
    if (little_endian != native_little_endian) { 
      byte temp;
      temp = p[0]; p[0] = p[3]; p[3] = temp;
      temp = p[1]; p[1] = p[2]; p[2] = temp;
    }
    depth[i] = *((float *) p);
  }
  fclose(f);
  free(data);
  return depth;
}
#endif
