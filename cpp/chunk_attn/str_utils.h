#pragma once

#ifdef USE_FMTLIB
#include <fmt/printf.h>
#else
#include <cstdarg>
#include <cstdlib>
#endif

#include <algorithm>
#include <codecvt>
#include <cstdint>
#include <locale>
#include <sstream>
#include <string>
#include <vector>


std::vector<std::string> split_str(const std::string& s, const char* delim = " \r\n\t", bool remove_empty = true);

std::string join_str(const std::vector<std::string>& tokens, const std::string& delim);

std::string& ltrim_str(std::string& str, const std::string& chars = "\t\n\v\f\r ");

std::string& rtrim_str(std::string& str, const std::string& chars = "\t\n\v\f\r ");

std::string& trim_str(std::string& str, const std::string& chars = "\t\n\v\f\r ");

bool str_ends_with(std::string const& str, std::string const& suffix);

std::string to_lower(const std::string& s);

#ifdef USE_FMTLIB
template <typename... Args>
std::string fmt_str(const char* fmt, Args&&... args) {
    return fmt::sprintf(fmt, std::forward<Args>(args)...);
}
#else
std::string fmt_str(const char* format, ...);
#endif

std::wstring str_to_wstr(const std::string& str);

std::string wstr_to_str(const std::wstring& wstr);

bool is_CJK(char32_t c);

bool is_accent(char32_t c);

char32_t strip_accent(char32_t c);

void replace_all(std::string& data, const std::string& str_to_search, const std::string& str_to_replace);

bool is_unicode_letter(const char32_t& ch);

bool is_unicode_number(const char32_t& ch);

bool is_unicode_seperator(const char32_t& ch);

bool is_unicode_space(const char32_t& ch);

bool not_category_LNZ(const char32_t& ch);
