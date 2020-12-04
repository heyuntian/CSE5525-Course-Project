#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cstdlib>
#include <unordered_map>
#include <set>
#include <ctime>
#include "parser.hpp"
#include "header.h"
#include "ioFile.h"
#include "strPackage.h"
#include "pbar.h"
#include "utils.h"

using namespace aria::csv;
using namespace std;
using namespace pbar;

#define MAX_STRING 100

int numWalks, walkLength;
char dataDir[MAX_STRING];
int movie_n, movie_base, genre_n, genre_base, cast_n, cast_base, user_n, user_base;
vector<double> dist(3);

vector<vector<uint32_t>> m2c;
vector<vector<uint32_t>> c2m;
vector<vector<uint32_t>> m2g;
vector<vector<uint32_t>> g2m;
vector<vector<uint32_t>> u2m;
vector<vector<double>> u2mr;
vector<vector<uint32_t>> m2u;
vector<vector<double>> m2ur;

// unordered_map<uint32_t, string> id2user;
// unordered_map<uint32_t, string> id2subforum;
// vector<uint32_t> postId;
// vector<uint32_t> commentId;
// unordered_map<uint32_t, uint32_t> msg2user; // post & comment to user
// vector<vector<uint32_t>> user2post;
// vector<vector<uint32_t>> user2comment;
// unordered_map<uint32_t, uint32_t> msg2subforum; // post & comment to subforum
// vector<vector<uint32_t>> subforum2post;
// unordered_map<uint32_t, uint32_t> comment2post; // comment to post
// unordered_map<uint32_t, vector<uint32_t>> post2comment;

int row_n, col_n, col, i;

void getArgs(int argc, char** argv) {
    numWalks = -1;
    walkLength = -1;
    if (argc == 1) {
        cout << "\t-dir <file>" << endl;
        cout << "\t-num_walks <int>, the number of random walks generated for each node" << endl;
        cout << "\t-length <int>, the length of each random walk" << endl;
        INFO("Error: Invalid parameters");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < argc; i++) {
        if (argv[i] == string("-dir")) {
            strcpy(dataDir, argv[i + 1]);
            if (dataDir[strlen(dataDir) - 1] != '/') {
                strcat(dataDir, "/");
            }
        }
        if (argv[i] == string("-num_walks")) {
            numWalks = atoi(argv[i + 1]);
        }
        if (argv[i] == string("-length")) {
            walkLength = atoi(argv[i + 1]);
        }
        if (numWalks == -1) {
            numWalks = 100;
        }
        if (walkLength == -1) {
            walkLength = 80;
        }
    }
    INFO("getArgs:\tfinished\n\tArguments received:", dataDir, numWalks, walkLength);
}

void readData() {
    INFO("readData: start");
    // read basic information
    string file_basic = dataDir + (string) "datainfo.md";
    ifstream f(file_basic);
    int read_totalN;
    f >> movie_n >> genre_n >> cast_n >> user_n >> read_totalN;
    f.close();
    INFO(movie_n, genre_n, cast_n, user_n, read_totalN);
    ASSERT(movie_n + genre_n + cast_n + user_n == read_totalN);
    movie_base = 0;
    genre_base = movie_base + movie_n;
    cast_base = genre_base + genre_n;
    user_base = cast_base + cast_n;
    INFO(movie_base, genre_base, cast_base, user_base);
    INFO("\tread graphInfo");

    // read mId2CC
    m2c.resize(movie_n);
    c2m.resize(cast_n);
    string file_mid2cc = dataDir + (string) "mId2CC.txt";
    f.open(file_mid2cc);
    readAttrs(&f, &m2c, &c2m, movie_base, cast_base, movie_n);
    f.close();
    INFO("\tread mId2CC");

    // read mId2Genre
    m2g.resize(movie_n);
    g2m.resize(genre_n);
    string file_mid2genre = dataDir + (string) "mId2Genre.txt";
    f.open(file_mid2genre);
    readAttrs(&f, &m2g, &g2m, movie_base, genre_base, movie_n);
    f.close();
    INFO("\tread mId2Genre");

    // read rating_train.csv
    m2u.resize(movie_n);
    m2ur.resize(movie_n);
    u2m.resize(user_n);
    u2mr.resize(user_n);
    string file_rating = dataDir + (string) "rating_train.csv";
    f.open(file_rating);
    CsvParser parser(f);
    string fields[4];
    int mId, uId; // , binary;
    double rating;
    bool f_firstline = true;
    for (auto& row : parser) {
        if (f_firstline) {
            f_firstline = false;
            continue;
        }
        col = 0;
        for (auto& field : row) {
            fields[col++] = field;
        }
        uId = atoi(fields[0].c_str()) - user_base;
        mId = atoi(fields[1].c_str()) - movie_base;
        ASSERT(uId >= 0 && uId < user_n);
        ASSERT(mId >= 0 && mId < movie_n);
        // binary = atoi(fields[2]);
        rating = atof(fields[3].c_str());
        u2m[uId].push_back(mId);
        u2mr[uId].push_back(rating);
        m2u[mId].push_back(uId);
        m2ur[mId].push_back(rating);
    }
    f.close();
    INFO("\tread rating_train");
}

void Walk(ofstream *f, uint32_t user) {
    // INFO("Walk:", user);
    vector<uint32_t> walk(walkLength + 1);
    walk[0] = user + user_base;
    double last_rating = -1;
    int currentLength = 0;
    uint32_t current = user;  // U
    uint32_t nextstep;
    uint8_t type = 0;
    uint32_t mId;
    uint8_t count = 0;
    bool f_m2c, f_m2g;
    while (currentLength < walkLength) {
        /* randomly pick a metapath
        0: U-M-U
        1: U-M-G-M-U
        2: U-M-C-M-U
        */
        // if (walkLength - currentLength < 4) {
        //     type = 0;
        // }
        // else {
        //     type = (uint8_t) wgtPick(dist);
        // }

        // count = 0;
        // while (true) {
        if (last_rating == -1) {
            nextstep = rand() % (u2m[current].size());
        }
        else {
            nextstep = softmaxPick(last_rating, u2mr[current]);
        }
        // mId = u2m[current][nextstep];

            // if ((type == 0) || \
            //     ((type == 1) && (m2g[mId].size() > 0)) || \
            //     ((type == 2) && (m2c[mId].size() > 0))) {
            //         break;
            //     }
            // else {
            //     INFO(type, current, mId, m2g[mId].size(), m2c[mId].size());
            //     INFO(walk);
            //     INFO(currentLength, walk[currentLength]);
            //     count++;
            //     ASSERT(count <= 5);
            // }
            // if (count == 5) {
            //     type = 0;
            // }
        // }
        last_rating = u2mr[current][nextstep];
        nextstep = u2m[current][nextstep];
        currentLength++;
        walk[currentLength] = nextstep + movie_base;
        current = nextstep;  // M


        f_m2c = (m2c[current].size() == 0);
        f_m2g = (m2g[current].size() == 0);
        if ((walkLength - currentLength < 3) || (f_m2g && f_m2c)) {
            type = 0;
        }
        else {
            type = (uint8_t) wgtPick(dist);
            if (f_m2c && (type == 2)) {
                type = rand() % 2;
            }
            else if (f_m2g && (type == 1)) {
                type = (rand() % 2) * 2;
            }
        }

        if (type == 1) {
            nextstep = UniformPick(m2g[current]);
            currentLength++;
            walk[currentLength] = nextstep + genre_base;
            current = nextstep;  // G

            count = 0;
            while (true) {
                nextstep = UniformPick(g2m[current]);
                if (m2u[nextstep].size() > 0) {
                    break;
                }
                else {
                    count++;
                }
                if (count == 8) {
                    nextstep = walk[currentLength-1] - movie_base;
                }
                // else {
                //     INFO(type, current, nextstep, m2u[nextstep].size());
                //     INFO(walk);
                //     INFO(currentLength, walk[currentLength]);
                //     count++;
                //     ASSERT(count < 8);
                // }
            }
            currentLength++;
            walk[currentLength] = nextstep + movie_base;
            current = nextstep;  // M
        }
        else if (type == 2) {
            nextstep = UniformPick(m2c[current]);
            currentLength++;
            walk[currentLength] = nextstep + cast_base;
            current = nextstep;  // C

            count = 0;
            while (true) {
                nextstep = UniformPick(c2m[current]);
                if (m2u[nextstep].size() > 0) {
                    break;
                }
                else {
                    count++;
                }
                if (count == 8) {
                    nextstep = walk[currentLength-1] - movie_base;
                }
                // else {
                //     INFO(type, current, nextstep, m2u[nextstep].size());
                //     INFO(walk);
                //     INFO(currentLength, walk[currentLength]);
                //     count++;
                //     ASSERT(count < 8);
                // }
            }
            currentLength++;
            walk[currentLength] = nextstep + movie_base;
            current = nextstep;  // M
        }

        // the last user
        nextstep = softmaxPick(last_rating, m2ur[current]);
        last_rating = m2ur[current][nextstep];
        nextstep = m2u[current][nextstep];
        currentLength++;
        walk[currentLength] = nextstep + user_base;
        current = nextstep;  // U
    }

    (*f) << walk[0];
    for (int i = 1; i < walkLength + 1; i++){
        (*f) << " " << walk[i];
    }

    (*f) << endl;
}

void generateWalks() {
    INFO("generateWalks: start");
    // check if dataDir exists, mkdir if not
    if (!isFolderExist(dataDir)) {
        int dummy = createDirectory(dataDir);
        INFO("generateWalks:\tdirectory", dataDir ,"created");
    }
    // create ofstream
    string identDir = slashToUnderscore(((string)dataDir).substr(0, strlen(dataDir) - 2));
    string outputFile = dataDir + (string) "walks.txt";
    ofstream fout(outputFile);
    // fout << user_n << " " << numWalks << " " << walkLength << endl;
    srand(time(NULL)); // set random seed
    // generate walks
    uint32_t j;
    vector<uint32_t> users(user_n);
    for (j = 0; j < user_n; j++) {
        users[j] = j;
    }
    for (uint32_t user=0; user<user_n; user++) {
        if (user % 100 == 0) {
            cout << user << endl;
        }
        if (u2m[user].size() == 0) {
            continue;
        }
        for (j = 0; j < numWalks; j++) {
            Walk(&fout, user);
        }
    }
    // close f
    INFO("generateWalks:\tfinished");
}

int main(int argc, char** argv) {
    srand(time(NULL));
    getArgs(argc, argv); // get arguments
    readData();
    dist[0] = 0.5;
    dist[1] = 0.1;
    dist[2] = 0.4;
    generateWalks();
    return 0;
}