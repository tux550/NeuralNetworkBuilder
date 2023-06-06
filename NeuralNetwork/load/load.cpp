#include "./load.h"

alg::vec_mat load_file(std::string filename) {
    std::ifstream DataFile(filename);
    std::string line;
    std::string num;
    char delim = ' '; 

    alg::vec_mat vec_data{};
    while (std::getline(DataFile, line)) {
        std::stringstream ss(line);
        alg::t_row tmp_row{};
        while (std::getline(ss, num, delim)) {
            tmp_row.push_back( std::stod(num) );
        }
        vec_data.push_back(alg::Matrix{{tmp_row}});
    }
    return vec_data;
}
