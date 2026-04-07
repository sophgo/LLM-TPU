//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// Distributed pipeline network helpers for Qwen3_VL
//
//===----------------------------------------------------------------------===//
#pragma once

#include <arpa/inet.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

//===------------------------------------------------------------===//
// Message Types
//===------------------------------------------------------------===//
enum MsgType : int32_t {
  MSG_PREFILL = 1,  // Prefill request
  MSG_DECODE = 2,   // Decode request
  MSG_TOKEN = 3,    // Token result
  MSG_CLEAR = 4,    // Clear history / reset
  MSG_SHUTDOWN = 5, // Shutdown
};

//===------------------------------------------------------------===//
// Step0 → Step1: Prefill metadata
//===------------------------------------------------------------===//
struct PrefillMeta {
  int32_t msg_type;            // MSG_PREFILL
  int32_t token_length;        // number of input tokens
  int32_t num_deepstack;       // deepstack buffers count (0 if !vit_run)
  int32_t position_ids_count;  // total ints in position_ids (ori_length * 3)
  int32_t hidden_size;         // HIDDEN_SIZE
  int32_t visited_token_count; // for sample_head repetition penalty
};
// Followed by variable data:
//   int32_t  position_ids[position_ids_count]
//   int32_t  visited_tokens[visited_token_count]
//   uint16_t hidden_states[token_length * hidden_size]
//   uint16_t deepstack[num_deepstack][token_length * hidden_size]

//===------------------------------------------------------------===//
// Step0 → Step1: Decode metadata
//===------------------------------------------------------------===//
struct DecodeMeta {
  int32_t msg_type;        // MSG_DECODE
  int32_t position_ids[3]; // 3D position ids for decode
  int32_t hidden_size;     // HIDDEN_SIZE
};
// Followed by:
//   uint16_t hidden_state[hidden_size]

//===------------------------------------------------------------===//
// Step1 → Step2: Hidden state for LMHead
//===------------------------------------------------------------===//
struct LMHeadMeta {
  int32_t msg_type;            // MSG_PREFILL or MSG_DECODE
  int32_t hidden_size;         // HIDDEN_SIZE
  int32_t visited_token_count; // for sampling; 0 for decode
};
// Followed by:
//   int32_t  visited_tokens[visited_token_count] (only for prefill)
//   uint16_t hidden_state[hidden_size]

//===------------------------------------------------------------===//
// Token result (Step2 → Step1, and Step1 → Step0)
//===------------------------------------------------------------===//
struct TokenMsg {
  int32_t token;
};

struct TokenWithHistory {
  int32_t token;
  int32_t history_length;
};

//===------------------------------------------------------------===//
// Simple command messages (clear / shutdown)
//===------------------------------------------------------------===//
struct SimpleMsg {
  int32_t msg_type;
};

//===------------------------------------------------------------===//
// TCP Helpers
//===------------------------------------------------------------===//

// Send exactly n bytes, returns false on error
static inline bool send_all(int fd, const void *buf, size_t n) {
  const char *p = static_cast<const char *>(buf);
  while (n > 0) {
    ssize_t sent = send(fd, p, n, MSG_NOSIGNAL);
    if (sent <= 0)
      return false;
    p += sent;
    n -= static_cast<size_t>(sent);
  }
  return true;
}

// Receive exactly n bytes, returns false on error/disconnect
static inline bool recv_all(int fd, void *buf, size_t n) {
  char *p = static_cast<char *>(buf);
  while (n > 0) {
    ssize_t got = recv(fd, p, n, 0);
    if (got <= 0)
      return false;
    p += got;
    n -= static_cast<size_t>(got);
  }
  return true;
}

// Create a TCP server socket listening on the given port
static inline int create_server(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    perror("socket");
    return -1;
  }
  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    perror("bind");
    close(fd);
    return -1;
  }
  if (listen(fd, 1) < 0) {
    perror("listen");
    close(fd);
    return -1;
  }
  printf("Listening on port %d ...\n", port);
  return fd;
}

// Accept one client connection
static inline int accept_client(int server_fd) {
  struct sockaddr_in client_addr;
  socklen_t len = sizeof(client_addr);
  int fd = accept(server_fd, (struct sockaddr *)&client_addr, &len);
  if (fd >= 0) {
    int opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    int bufsize = 16 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
    printf("Accepted connection from %s:%d\n", ip, ntohs(client_addr.sin_port));
  }
  return fd;
}

// Connect to a TCP server, retries until success
static inline int connect_to(const std::string &host, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  assert(fd >= 0);
  int opt = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
  int bufsize = 16 * 1024 * 1024;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

  while (true) {
    int ret = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (ret == 0)
      break;
    printf("Connecting to %s:%d ... retrying\n", host.c_str(), port);
    sleep(1);
  }
  printf("Connected to %s:%d\n", host.c_str(), port);
  return fd;
}
