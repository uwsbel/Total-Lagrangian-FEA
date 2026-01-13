#include "cli_utils.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ANCFCPUUtils {
namespace {

std::string ToLower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

std::string_view TrimLeftDashes(std::string_view s) {
  while (!s.empty() && s.front() == '-') {
    s.remove_prefix(1);
  }
  return s;
}

}  // namespace

Cli::Cli(std::string program_name) : program_name_(std::move(program_name)) {}

void Cli::SetDescription(std::string description) {
  description_ = std::move(description);
}

void Cli::AddBool(std::string name, bool default_value, std::string help) {
  AddOption(std::move(name), Type::kBool, Value{default_value}, std::move(help),
            /*allow_missing_value=*/true);
}

void Cli::AddInt(std::string name, int default_value, std::string help) {
  AddOption(std::move(name), Type::kInt, Value{default_value}, std::move(help),
            /*allow_missing_value=*/false);
}

void Cli::AddDouble(std::string name, double default_value, std::string help) {
  AddOption(std::move(name), Type::kDouble, Value{default_value},
            std::move(help), /*allow_missing_value=*/false);
}

void Cli::AddString(std::string name, std::string default_value,
                    std::string help) {
  AddOption(std::move(name), Type::kString, Value{std::move(default_value)},
            std::move(help), /*allow_missing_value=*/false);
}

void Cli::AddOptionalString(std::string name, std::string default_value,
                            std::string help) {
  AddOption(std::move(name), Type::kString, Value{std::move(default_value)},
            std::move(help), /*allow_missing_value=*/true);
}

void Cli::AddOption(std::string name, Type type, Value default_value,
                    std::string help, bool allow_missing_value) {
  if (name.empty()) {
    throw std::invalid_argument("Cli::AddOption: name must be non-empty");
  }
  if (name[0] == '-') {
    throw std::invalid_argument("Cli::AddOption: name must not start with '-'");
  }
  if (options_.find(name) != options_.end()) {
    throw std::invalid_argument("Cli::AddOption: duplicate option: " + name);
  }

  Option opt;
  opt.type                = type;
  opt.help                = std::move(help);
  opt.default_value       = default_value;
  opt.value               = default_value;
  opt.allow_missing_value = allow_missing_value;
  options_.emplace(name, std::move(opt));
  option_order_.push_back(std::move(name));
}

bool Cli::Parse(int argc, char** argv, std::string* error) {
  help_requested_ = false;
  for (auto& [_, opt] : options_) {
    opt.value  = opt.default_value;
    opt.is_set = false;
  }

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i] ? argv[i] : "");
    if (arg.empty()) {
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      help_requested_ = true;
      continue;
    }
    const auto err = ParseOneArg(arg);
    if (err.has_value()) {
      if (error)
        *error = *err;
      return false;
    }
  }
  return true;
}

bool Cli::HelpRequested() const {
  return help_requested_;
}

bool Cli::IsSet(std::string_view name) const {
  return FindOptionOrDie(name).is_set;
}

bool Cli::GetBool(std::string_view name) const {
  const auto& opt = FindOptionOrDie(name);
  if (opt.type != Type::kBool) {
    throw std::logic_error("Cli::GetBool: option is not bool: " +
                           std::string(name));
  }
  return std::get<bool>(opt.value);
}

int Cli::GetInt(std::string_view name) const {
  const auto& opt = FindOptionOrDie(name);
  if (opt.type != Type::kInt) {
    throw std::logic_error("Cli::GetInt: option is not int: " +
                           std::string(name));
  }
  return std::get<int>(opt.value);
}

double Cli::GetDouble(std::string_view name) const {
  const auto& opt = FindOptionOrDie(name);
  if (opt.type != Type::kDouble) {
    throw std::logic_error("Cli::GetDouble: option is not double: " +
                           std::string(name));
  }
  return std::get<double>(opt.value);
}

std::string Cli::GetString(std::string_view name) const {
  const auto& opt = FindOptionOrDie(name);
  if (opt.type != Type::kString) {
    throw std::logic_error("Cli::GetString: option is not string: " +
                           std::string(name));
  }
  return std::get<std::string>(opt.value);
}

std::optional<std::string> Cli::ParseOneArg(std::string_view arg) {
  if (arg == "--") {
    return std::string("Unexpected positional args: '--' is not supported");
  }
  if (arg.size() < 3 || arg[0] != '-' || arg[1] != '-') {
    return "Unknown argument (expected --key or --key=value): " +
           std::string(arg);
  }

  std::string_view body = TrimLeftDashes(arg);
  if (body.empty()) {
    return "Invalid argument: " + std::string(arg);
  }

  const size_t eq = body.find('=');
  if (eq == std::string_view::npos) {
    return SetOptionFromString(body, std::nullopt);
  }

  const std::string_view name  = body.substr(0, eq);
  const std::string_view value = body.substr(eq + 1);
  if (name.empty()) {
    return "Invalid argument (empty key): " + std::string(arg);
  }
  return SetOptionFromString(name, value);
}

std::optional<std::string> Cli::SetOptionFromString(
    std::string_view name, std::optional<std::string_view> value) {
  auto it = options_.find(std::string(name));
  if (it == options_.end()) {
    return "Unknown option: --" + std::string(name);
  }
  Option& opt = it->second;

  if (opt.type == Type::kBool) {
    if (!value.has_value()) {
      opt.value  = true;
      opt.is_set = true;
      return std::nullopt;
    }
    const auto b = ParseBool(*value);
    if (!b.has_value()) {
      return "Invalid bool for --" + std::string(name) + ": " +
             std::string(*value);
    }
    opt.value  = *b;
    opt.is_set = true;
    return std::nullopt;
  }

  if (!value.has_value()) {
    if (opt.allow_missing_value) {
      opt.value  = opt.default_value;
      opt.is_set = true;
      return std::nullopt;
    }
    return "Missing value for --" + std::string(name) +
           TypeSuffix(opt.type, /*allow_missing_value=*/false);
  }

  if (opt.type == Type::kInt) {
    const auto v = ParseInt(*value);
    if (!v.has_value()) {
      return "Invalid int for --" + std::string(name) + ": " +
             std::string(*value);
    }
    opt.value  = *v;
    opt.is_set = true;
    return std::nullopt;
  }

  if (opt.type == Type::kDouble) {
    const auto v = ParseDouble(*value);
    if (!v.has_value()) {
      return "Invalid double for --" + std::string(name) + ": " +
             std::string(*value);
    }
    opt.value  = *v;
    opt.is_set = true;
    return std::nullopt;
  }

  if (opt.type == Type::kString) {
    opt.value  = std::string(*value);
    opt.is_set = true;
    return std::nullopt;
  }

  return "Internal error: unhandled option type for --" + std::string(name);
}

std::string Cli::TypeSuffix(Type type, bool allow_missing_value) {
  const char* suffix = "";
  switch (type) {
    case Type::kBool:
      suffix = "";
      break;
    case Type::kInt:
      suffix = "=<int>";
      break;
    case Type::kDouble:
      suffix = "=<double>";
      break;
    case Type::kString:
      suffix = "=<string>";
      break;
  }
  if (!allow_missing_value || type == Type::kBool) {
    return suffix;
  }
  // Convert "=<type>" to "[=<type>]" for optional values.
  return std::string("[") + suffix + "]";
}

std::optional<bool> Cli::ParseBool(std::string_view s) {
  const std::string v = ToLower(s);
  if (v == "1" || v == "true" || v == "t" || v == "yes" || v == "y" ||
      v == "on") {
    return true;
  }
  if (v == "0" || v == "false" || v == "f" || v == "no" || v == "n" ||
      v == "off") {
    return false;
  }
  return std::nullopt;
}

std::optional<int> Cli::ParseInt(std::string_view s) {
  try {
    size_t idx  = 0;
    const int v = std::stoi(std::string(s), &idx);
    if (idx != s.size()) {
      return std::nullopt;
    }
    return v;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<double> Cli::ParseDouble(std::string_view s) {
  try {
    size_t idx     = 0;
    const double v = std::stod(std::string(s), &idx);
    if (idx != s.size()) {
      return std::nullopt;
    }
    return v;
  } catch (...) {
    return std::nullopt;
  }
}

std::string Cli::ValueToString(const Value& v) {
  if (std::holds_alternative<bool>(v)) {
    return std::get<bool>(v) ? "true" : "false";
  }
  if (std::holds_alternative<int>(v)) {
    return std::to_string(std::get<int>(v));
  }
  if (std::holds_alternative<double>(v)) {
    std::ostringstream oss;
    oss << std::setprecision(16) << std::get<double>(v);
    return oss.str();
  }
  return std::get<std::string>(v);
}

Cli::Option& Cli::FindOptionOrDie(std::string_view name) {
  auto it = options_.find(std::string(name));
  if (it == options_.end()) {
    throw std::out_of_range("Cli: unknown option: " + std::string(name));
  }
  return it->second;
}

const Cli::Option& Cli::FindOptionOrDie(std::string_view name) const {
  auto it = options_.find(std::string(name));
  if (it == options_.end()) {
    throw std::out_of_range("Cli: unknown option: " + std::string(name));
  }
  return it->second;
}

void Cli::PrintUsage(std::ostream& os) const {
  os << "Usage: " << program_name_ << " [options]\n";
  if (!description_.empty()) {
    os << "\n" << description_ << "\n";
  }

  os << "\nOptions:\n";

  size_t width = 0;
  for (const auto& name : option_order_) {
    const auto& opt = options_.at(name);
    const std::string flag =
        "--" + name + TypeSuffix(opt.type, opt.allow_missing_value);
    width = std::max(width, flag.size());
  }
  width = std::max<size_t>(width, std::string("--help").size());
  width = std::min<size_t>(width, 40);

  for (const auto& name : option_order_) {
    const auto& opt = options_.at(name);
    const std::string flag =
        "--" + name + TypeSuffix(opt.type, opt.allow_missing_value);
    os << "  " << std::left << std::setw(static_cast<int>(width) + 2) << flag
       << opt.help;
    os << " (default: " << ValueToString(opt.default_value) << ")";
    os << "\n";
  }
  os << "  " << std::left << std::setw(static_cast<int>(width) + 2) << "--help"
     << "show this help message\n";
}

}  // namespace ANCFCPUUtils
