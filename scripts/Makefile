# Makefile for grail-upload-monitor

CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lpthread -lcurl

TARGET = grail-upload-monitor
SOURCE = grail-upload-monitor.c

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	@echo ""
	@echo "Build complete: $(TARGET)"
	@echo ""
	@echo "To install:"
	@echo "  sudo cp $(TARGET) /usr/local/bin/"
	@echo ""
	@echo "To run:"
	@echo "  ./$(TARGET)"
	@echo ""
	@echo "Don't forget to create /root/.grail-smtp.conf with your credentials!"

clean:
	rm -f $(TARGET)

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
	chmod 755 /usr/local/bin/$(TARGET)
	@echo "Installed to /usr/local/bin/$(TARGET)"
