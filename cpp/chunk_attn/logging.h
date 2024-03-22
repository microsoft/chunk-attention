#pragma once

#include <vector>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

class Logger {
  public:
    explicit Logger(bool std_out = false, bool std_err = true,
                    const spdlog::level::level_enum& level = spdlog::level::level_enum::debug,
                    const std::string& format = "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%P:%t] %v");

    template <typename... Rest>
    void Debug(const char* fmt, const Rest&... rest) {
        inner_logger_->debug(fmt, rest...);
    }

    template <typename... Rest>
    void Info(const char* fmt, const Rest&... rest) {
        inner_logger_->info(fmt, rest...);
    }

    template <typename... Rest>
    void Warn(const char* fmt, const Rest&... rest) {
        inner_logger_->warn(fmt, rest...);
    }

    template <typename... Rest>
    void Error(const char* fmt, const Rest&... rest) {
        inner_logger_->error(fmt, rest...);
    }

    void Debug(const char* msg);
    void Info(const char* msg);
    void Warn(const char* msg);
    void Error(const char* msg);

  private:
    void AddConsole(bool std_out = false, bool std_err = true);

    std::string format_;
    std::vector<std::shared_ptr<spdlog::sinks::sink>> sinks_;
    std::shared_ptr<spdlog::logger> inner_logger_;
    spdlog::level::level_enum level_;
};

extern Logger logger;
//#define LOG_DEBUG(...) logger.Debug(__VA_ARGS__)
#define LOG_DEBUG(...) do{} while(0)
#define LOG_INFO(...) logger.Info(__VA_ARGS__)
#define LOG_WARN(...) logger.Warn(__VA_ARGS__)
#define LOG_ERROR(...) logger.Error(__VA_ARGS__)
