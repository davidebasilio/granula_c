.PHONY: all clean

CFLAGS  = -Iinclude -I../../include -O3 -std=gnu99 -fopenmp -ggdb3
LDFLAGS =

SRCS = $(shell find src -name '*.c')
OBJS = $(SRCS:src/%.c=bin/%.o)

TARGET = kmeans

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(CFLAGS) $(LDFLAGS)

bin/%.o: src/%.c
	@mkdir -p `dirname $@`
	$(CC) $(CFLAGS) -MMD -c $< -o $@

clean:
	-rm -f $(OBJS) $(OBJS:%.o=%.d) $(TARGET)
	-rm -rf bin

-include $(OBJS:bin/%.o=bin/%.d)

