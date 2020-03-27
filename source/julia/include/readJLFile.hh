#ifndef READJLFILE_H
#define READJLFILE_H

#include <stdio.h>
#include <julia.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#define FILE_OK 0
#define FILE_NOT_EXIST 1
#define FILE_TO_LARGE 2
#define FILE_READ_ERROR 3

char *readFile(char *filename);
void readJLFile(char *filename);

#endif
