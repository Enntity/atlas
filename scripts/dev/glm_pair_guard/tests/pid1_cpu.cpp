// SPDX-License-Identifier: AGPL-3.0-only
// Local CPU qualification only. Never link or run this with a model/GPU.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <vector>

static void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}
static uint64_t now_ms() {
    timespec t{};
    require(clock_gettime(CLOCK_BOOTTIME, &t) == 0, "BOOTTIME");
    return uint64_t(t.tv_sec) * 1000 + uint64_t(t.tv_nsec) / 1000000;
}
struct Fd {
    int value = -1;
    Fd() = default;
    explicit Fd(int v) : value(v) { require(v >= 0, "descriptor"); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& other) noexcept : value(other.value) { other.value = -1; }
    Fd& operator=(Fd&& other) noexcept {
        if (value >= 0) close(value);
        value = other.value; other.value = -1; return *this;
    }
    ~Fd() { if (value >= 0) close(value); }
};
static std::string read_file(const std::string& path) {
    Fd fd(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    std::array<char, 16385> bytes{};
    ssize_t n = read(fd.value, bytes.data(), bytes.size());
    require(n >= 0 && size_t(n) < bytes.size(), "bounded file read");
    return std::string(bytes.data(), size_t(n));
}
static void receipt(const std::string& path, const std::string& data) {
    Fd fd(open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
    require(write(fd.value, data.data(), data.size()) == ssize_t(data.size()), "receipt write");
}
static bool exists(const std::string& path) {
    struct stat s{};
    if (lstat(path.c_str(), &s) == 0) return true;
    require(errno == ENOENT, "witness lstat"); return false;
}
template<class Predicate> static void until(uint64_t deadline, Predicate predicate) {
    while (!predicate()) {
        require(now_ms() < deadline, "bounded wait expired");
        int rc = poll(nullptr, 0, 5);
        require(rc >= 0 || errno == EINTR, "bounded wait poll");
    }
}
static bool dead(int fd) {
    pollfd p{fd, POLLIN, 0};
    int rc = poll(&p, 1, 0);
    require(rc >= 0 || errno == EINTR, "pidfd poll");
    require(!(p.revents & POLLNVAL), "invalid pidfd");
    return (p.revents & POLLIN) != 0;
}
struct Process {
    Fd fd;
    pid_t host = -1;
    bool owned = false, reaped = false;
};
static void signal_kill(Process& p) {
    require(p.fd.value >= 0, "kill requires retained pidfd");
    int rc = int(syscall(SYS_pidfd_send_signal, p.fd.value, SIGKILL, nullptr, 0));
    require(rc == 0 || errno == ESRCH, "pidfd SIGKILL");
}
static void await_death(Process& p, uint64_t deadline) {
    until(deadline, [&] { return dead(p.fd.value); });
}
static siginfo_t reap(Process& p) {
    require(p.owned && !p.reaped, "reap requires unreaped direct child");
    siginfo_t info{};
    require(waitid(P_PIDFD, id_t(p.fd.value), &info, WEXITED | WNOHANG) == 0,
            "pidfd waitid");
    require(info.si_pid == p.host, "waitid identity");
    p.reaped = true; return info;
}
static void nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL);
    require(flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0, "nonblocking");
}
// One finite wire transfer, never a read-until-close or unbounded drain.
static void transfer(int fd, std::array<uint8_t, 112>& frame, bool send_frame) {
    size_t offset = 0;
    uint64_t end = now_ms() + 3000;
    while (offset < frame.size()) {
        require(now_ms() < end, "control transfer deadline");
        pollfd p{fd, short(send_frame ? POLLOUT : POLLIN), 0};
        int rc = poll(&p, 1, 20);
        require(rc >= 0 || errno == EINTR, "control poll");
        require(!(p.revents & (POLLNVAL | POLLERR | POLLHUP)), "control closed");
        if (!(p.revents & p.events)) continue;
        ssize_t n = send_frame
            ? send(fd, frame.data() + offset, frame.size() - offset, MSG_NOSIGNAL)
            : recv(fd, frame.data() + offset, frame.size() - offset, 0);
        if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
        require(n > 0, "control transfer"); offset += size_t(n);
    }
    require(now_ms() < end, "control completion deadline");
}
static std::string proc(pid_t pid, const char* tail) {
    return "/proc/" + std::to_string(pid) + tail;
}
static ino_t namespace_inode(pid_t host) {
    struct stat s{};
    require(stat(proc(host, "/ns/pid").c_str(), &s) == 0, "PID namespace stat");
    return s.st_ino;
}
static int namespace_pid(pid_t host) {
    std::istringstream lines(read_file(proc(host, "/status")));
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind("NSpid:", 0) != 0) continue;
        std::istringstream values(line.substr(6));
        int value = -1, last = -1;
        while (values >> value) last = value;
        require(last > 0, "NSpid value"); return last;
    }
    throw std::runtime_error("NSpid absent");
}
static pid_t only_child(pid_t parent) {
    auto text = read_file(proc(parent, ("/task/" + std::to_string(parent) + "/children").c_str()));
    std::istringstream ids(text);
    pid_t first = -1, extra = -1;
    require(bool(ids >> first) && first > 0 && !(ids >> extra), "exact single child relationship");
    return first;
}
static void observe_child(Process& result, Process& parent) {
    require(!dead(parent.fd.value), "parent alive before child observation");
    result.host = only_child(parent.host);
    result.fd = Fd(int(syscall(SYS_pidfd_open, result.host, 0)));
    require(only_child(parent.host) == result.host && !dead(parent.fd.value) &&
            !dead(result.fd.value), "stable child observation");
}
// Prepared argv and single-threaded fork only; no C++ allocation/Drop in child.
static void spawn(Process& p, std::vector<std::string> args, int control = -1,
                  bool namespace_init = false) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    int pipes[2]; require(pipe2(pipes, O_CLOEXEC) == 0, "startup pipe");
    Fd read_gate(pipes[0]), write_gate(pipes[1]);
    int readiness[2]; require(pipe2(readiness, O_CLOEXEC) == 0, "startup readiness pipe");
    Fd read_ready(readiness[0]), write_ready(readiness[1]);
    // Namespace PID1 sees an outside-namespace parent as0. A readiness/gate
    // exchange also closes the pre-PR_SET_PDEATHSIG parent-death race here.
    pid_t parent = namespace_init ? 0 : getpid();
    pid_t pid = fork(); require(pid >= 0, "fork");
    if (pid == 0) {
        close(write_gate.value);
        close(read_ready.value);
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(80);
        if (write(write_ready.value, "R", 1) != 1) _exit(86);
        close(write_ready.value);
        char byte = 0;
        ssize_t n;
        do { n = read(read_gate.value, &byte, 1); } while (n < 0 && errno == EINTR);
        if (n != 1 || byte != 'X' || getppid() != parent) _exit(81);
        close(read_gate.value);
        if (syscall(SYS_close_range, 3u, ~0u, CLOSE_RANGE_CLOEXEC) != 0) _exit(82);
        if (control >= 0 && (dup2(control, 3) < 0 || fcntl(3, F_SETFD, 0) < 0)) _exit(83);
        execv(argv[0], argv.data()); _exit(84);
    }
    p.host = pid; p.owned = true;
    // On pidfd failure closing the unreleased pipe prevents exec; no PID kill fallback.
    p.fd = Fd(int(syscall(SYS_pidfd_open, pid, 0)));
    write_ready = Fd(); read_gate = Fd(); nonblocking(read_ready.value);
    until(now_ms() + 3000, [&] {
        char ready = 0;
        ssize_t count = read(read_ready.value, &ready, 1);
        if (count < 0 && (errno == EAGAIN || errno == EINTR)) return false;
        require(count == 1 && ready == 'R', "child parent-death setup readiness");
        return true;
    });
    require(write(write_gate.value, "X", 1) == 1, "startup release");
}

static std::string at_exit_path;
static void at_exit_witness() {
    int fd = open(at_exit_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd >= 0) { if (write(fd, "atexit", 6) != 6) _exit(87); close(fd); }
}
struct Witness {
    std::string path;
    ~Witness() {
        int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
        if (fd >= 0) { if (write(fd, "drop", 4) != 4) _exit(87); close(fd); }
    }
};
static int probe(const std::string& mode, const std::string& dir) {
    require(mode == "healthy" || mode == "decoy" || mode == "namespace", "probe mode");
    std::string role = mode;
    if (mode == "namespace") {
        int protection = 0;
        require(prctl(PR_GET_PDEATHSIG, &protection) == 0 && protection == SIGKILL,
                "guard child parent-death protection");
        pid_t leaf = fork(); require(leaf >= 0, "witness descendant fork");
        if (leaf == 0) role = "descendant";
        else role = "child";
    }
    int protection = -1;
    require(prctl(PR_GET_PDEATHSIG, &protection) == 0, "probe parent-death query");
    if (role == "descendant") require(protection == 0, "descendant must lack PDEATHSIG");
    at_exit_path = dir + "/" + role + ".atexit";
    require(std::atexit(at_exit_witness) == 0, "atexit registration");
    Witness witness{dir + "/" + role + ".drop"};
    receipt(dir + "/" + role + ".ready", "pid=" + std::to_string(getpid()) +
            " pdeath=" + std::to_string(protection) + "\n");
    if (mode == "healthy") return 0;
    // A bounded harmless CPU wait; natural exit would create forbidden witnesses.
    until(now_ms() + 60000, [] { return false; });
    return 1;
}

static int qualification(const std::string& guard) {
    std::array<Process, 5> owners; // healthy, decoy, init, guarded child, descendant
    std::string dir;
    try {
        struct sigaction action{};
        sigemptyset(&action.sa_mask); action.sa_handler = SIG_IGN;
        require(sigaction(SIGPIPE, &action, nullptr) == 0, "ignore harness pipe failure signal");
        action.sa_handler = SIG_DFL;
        require(sigaction(SIGCHLD, &action, nullptr) == 0, "own direct child reaping");
        require(!guard.empty() && guard[0] == '/', "absolute frozen guard ELF path required");
        char self_buffer[4096];
        ssize_t n = readlink("/proc/self/exe", self_buffer, sizeof(self_buffer));
        require(n > 0 && n < ssize_t(sizeof(self_buffer)), "own executable path");
        std::string self(self_buffer, size_t(n));
        char temp[] = "/tmp/atlas-pid1-cpu-XXXXXX";
        require(mkdtemp(temp) != nullptr, "private witness directory"); dir = temp;
        std::printf("receipts=%s\n", dir.c_str()); std::fflush(stdout);
        spawn(owners[0], {self, "--probe", "healthy", dir});
        await_death(owners[0], now_ms() + 5000);
        auto healthy = reap(owners[0]);
        require(healthy.si_code == CLD_EXITED && healthy.si_status == 0, "healthy probe exit0");
        require(read_file(dir + "/healthy.drop") == "drop" &&
                read_file(dir + "/healthy.atexit") == "atexit", "real cleanup witness controls");
        spawn(owners[1], {self, "--probe", "decoy", dir});
        until(now_ms() + 3000, [&] { return exists(dir + "/decoy.ready"); });
        ino_t outside = namespace_inode(getpid());
        require(namespace_inode(owners[1].host) == outside, "decoy outside namespace");
        int sockets[2]; require(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) == 0,
                                "control socketpair");
        Fd control(sockets[0]), inherited(sockets[1]); nonblocking(control.value);
        require(unshare(CLONE_NEWPID) == 0, "private PID namespace prerequisite");
        spawn(owners[2], {guard, "3", "10000", "20000", "10000", "5000", "30000", "10", "1000",
                         self, "--probe", "namespace", dir}, inherited.value, true);
        inherited = Fd();
        auto& init = owners[2];
        require(namespace_pid(init.host) == 1, "guard is actual namespace PID1");
        ino_t inside = namespace_inode(init.host);
        require(inside != outside && namespace_inode(getpid()) == outside, "distinct child PID namespace");
        std::array<uint8_t, 112> hello{}; transfer(control.value, hello, false);
        require(hello[0] == 0 && hello[1] == 0 && hello[2] == 0 && hello[3] == 108 &&
                hello[4] == 1 && hello[5] == 1 && hello[6] == 0 && hello[7] == 0,
                "exact guard HELLO schema");
        for (size_t i = 72; i < 80; ++i) require(hello[i] == 0, "initial ordinal zero");
        observe_child(owners[3], init);
        require(namespace_inode(owners[3].host) == inside, "gated child namespace");
        require(!exists(dir + "/child.ready") && !exists(dir + "/descendant.ready"), "pre-exec gate");
        require(!dead(init.fd.value) && !dead(owners[3].fd.value) && !dead(owners[1].fd.value),
                "all pre-START identities alive");
        hello[5] = 2; transfer(control.value, hello, true); // Echo full random identity/challenge.
        until(now_ms() + 3000, [&] { return exists(dir + "/child.ready") && exists(dir + "/descendant.ready"); });
        observe_child(owners[4], owners[3]);
        require(namespace_inode(owners[4].host) == inside && namespace_pid(owners[4].host) > 1,
                "descendant inside guarded namespace");
        require(read_file(dir + "/descendant.ready").find(" pdeath=0\n") != std::string::npos,
                "actual descendant has no parent-death signal");
        require(!dead(init.fd.value) && !dead(owners[3].fd.value) && !dead(owners[4].fd.value),
                "three namespace identities alive before PID1 kill");
        receipt(dir + "/identities", "outside=" + std::to_string(outside) + " inside=" +
                std::to_string(inside) + " init=" + std::to_string(init.host) + " child=" +
                std::to_string(owners[3].host) + " descendant=" + std::to_string(owners[4].host) + "\n");
        signal_kill(init);
        uint64_t end = now_ms() + 5000;
        for (size_t i : {2u, 3u, 4u}) await_death(owners[i], end);
        auto status = reap(init);
        require(status.si_code == CLD_KILLED && status.si_status == SIGKILL, "PID1 killed by SIGKILL");
        require(!dead(owners[1].fd.value), "outside decoy survives namespace death");
        for (const char* role : {"child", "descendant"}) {
            require(!exists(dir + "/" + role + ".drop") && !exists(dir + "/" + role + ".atexit"),
                    "namespace termination ran graceful cleanup");
        }
        signal_kill(owners[1]); await_death(owners[1], now_ms() + 3000); (void)reap(owners[1]);
        receipt(dir + "/PASS", "actual guard PID1 death; no-PDEATHSIG descendant dead; decoy survived\n");
        std::puts("PASS: local PID1 namespace death; no Docker/GPU/T2 claim");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s; retained receipts=%s\n", e.what(), dir.c_str());
        // Only retained exact pidfds. Init is killed first to contain its namespace.
        for (size_t i : {2u, 3u, 4u, 1u, 0u}) {
            auto& p = owners[i];
            if (p.fd.value < 0 || p.reaped) continue;
            try { signal_kill(p); await_death(p, now_ms() + 1500); if (p.owned) (void)reap(p); }
            catch (const std::exception& cleanup) { std::fprintf(stderr, "cleanup failed: %s\n", cleanup.what()); }
        }
        return 1;
    }
}
int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--probe") == 0) {
        try { return probe(argv[2], argv[3]); }
        catch (const std::exception&) { _exit(85); }
    }
    if (argc != 3 || std::strcmp(argv[1], "--run") != 0) {
        std::fprintf(stderr, "usage: pid1_cpu --run /absolute/frozen/guard-ELF\n"); return 2;
    }
    return qualification(argv[2]);
}
