#include "./load.h"

alg::vec_mat load_file(std::string filename) {
    std::ifstream DataFile(filename);
    std::string line;
    std::string num;
    char delim = ' '; 

    alg::vec_mat vec_data{};

    while (std::getline(DataFile, line)) {
        std::stringstream ss(line);
        alg::t_inprow tmp{};
        while (std::getline(ss, num, delim)) {
            tmp.push_back( std::stod(num) );
        }
        // Init arma mat from vector
        alg::t_mat single_data(&tmp.front(), 1, tmp.size());
        vec_data.push_back(single_data);
    }

    return vec_data;
}
