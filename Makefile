CXX = g++
CXXFLAGS = -O2 -std=c++17 -pthread -Wall -Wextra -march=native -I src
LDFLAGS = -lm

SRC_DIR = src
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/kdtree.cpp \
       $(SRC_DIR)/file_io.cpp \
       $(SRC_DIR)/thread_pool.cpp \
       $(SRC_DIR)/named_pipe.cpp

TARGET = knn_search
WORKER = knn_worker

all: $(TARGET) $(WORKER)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

$(WORKER): $(SRC_DIR)/worker.cpp $(SRC_DIR)/kdtree.cpp $(SRC_DIR)/file_io.cpp $(SRC_DIR)/named_pipe.cpp
	$(CXX) $(CXXFLAGS) $^ -o $(WORKER) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(WORKER) *.o *.bin *.dat