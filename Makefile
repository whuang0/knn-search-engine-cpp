CXX = g++
CXXFLAGS = -std=c++17 -O2 -pthread -Wall

SRC_DIR = src
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/kdtree.cpp \
       $(SRC_DIR)/file_io.cpp \
       $(SRC_DIR)/thread_pool.cpp \
       $(SRC_DIR)/named_pipe.cpp \
       $(SRC_DIR)/worker.cpp

TARGET = knn_search

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o