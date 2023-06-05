
#include "./logger.h"
void debug_print(std::string message) {
    #ifdef DEBUG
    std::cout << "[D]:" << message << std::endl;
    #endif
}

void info_print(std::string message) {
    #ifdef INFO
    std::cout << "[I]:" << message << std::endl;
    #endif
}

void verbose_print(std::string message) {
    #ifdef VERBOSE
    std::cout << "[V]:" << message << std::endl;
    #endif
}