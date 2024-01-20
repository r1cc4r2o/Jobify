GREEN := "\033[1;32m"
YELLOW := "\033[1;33m"
RED := "\033[1;31m"
NC := "\033[0m"

.PHONY: install run help

default: help

help:
	@echo ${GREEN}"Available targets:"${NC}
	@echo ${YELLOW}"  make install"${NC}" - Install dependencies."
	@echo ${YELLOW}"  make run"${NC}"     - Launch Rasa shell."
	@echo ${YELLOW}"  make help"${NC}"    - Show this help message."
	@echo ${GREEN}"Usage:"${NC}
	@echo "  make <target>"

install:
	@which python || (echo ${RED}"[!] Error: Python is not installed. Please install Python before proceeding."${NC} && exit 1)
	@echo ${YELLOW}"Installing dependencies..."${NC}
	@pip install -r requirements.txt
	@echo ${GREEN}"Dependencies installed successfully."${NC}
	@echo ${YELLOW}"Creating your vector database..."${NC}
	@python ./installation/install.py
	@echo ${GREEN}"Vector database created successfully."${NC}

run:
	@echo ${YELLOW}"Launching Rasa shell..."${NC}
	@rasa shell
	@echo ${GREEN}"Rasa shell closed."${NC}