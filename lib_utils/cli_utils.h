#pragma once

#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ANCFCPUUtils {

// Small CLI helper for demos:
// - Register expected options (type + default + help)
// - Parse argv of the form:
//     --key=value
//     --flag            (bool only)
//     --flag=true|false (bool only)
//     --help / -h
//
// Intended usage:
//   Cli cli(argv[0]);
//   cli.AddInt("steps", 50, "number of Solve() calls");
//   ...
//   std::string err;
//   if (!cli.Parse(argc, argv, &err) || cli.HelpRequested()) { ... }
class Cli {
 public:
  explicit Cli(std::string program_name);

  void SetDescription(std::string description);

  void AddBool(std::string name, bool default_value, std::string help);
  void AddInt(std::string name, int default_value, std::string help);
  void AddDouble(std::string name, double default_value, std::string help);
  void AddString(std::string name, std::string default_value, std::string help);
  // Allows both `--name` and `--name=VALUE` (useful for `--csv[=PATH]`).
  void AddOptionalString(std::string name, std::string default_value,
                         std::string help);

  bool Parse(int argc, char** argv, std::string* error = nullptr);
  bool HelpRequested() const;

  bool IsSet(std::string_view name) const;

  bool GetBool(std::string_view name) const;
  int GetInt(std::string_view name) const;
  double GetDouble(std::string_view name) const;
  std::string GetString(std::string_view name) const;

  void PrintUsage(std::ostream& os) const;

 private:
  enum class Type { kBool, kInt, kDouble, kString };

  using Value = std::variant<bool, int, double, std::string>;

  struct Option {
    Type type;
    std::string help;
    Value default_value;
    Value value;
    bool is_set              = false;
    bool allow_missing_value = false;
  };

  void AddOption(std::string name, Type type, Value default_value,
                 std::string help, bool allow_missing_value);

  std::optional<std::string> ParseOneArg(std::string_view arg);
  std::optional<std::string> SetOptionFromString(
      std::string_view name, std::optional<std::string_view> value);

  static std::string TypeSuffix(Type type, bool allow_missing_value);
  static std::optional<bool> ParseBool(std::string_view s);
  static std::optional<int> ParseInt(std::string_view s);
  static std::optional<double> ParseDouble(std::string_view s);
  static std::string ValueToString(const Value& v);

  Option& FindOptionOrDie(std::string_view name);
  const Option& FindOptionOrDie(std::string_view name) const;

  std::string program_name_;
  std::string description_;
  bool help_requested_ = false;

  std::unordered_map<std::string, Option> options_;
  std::vector<std::string> option_order_;
};

}  // namespace ANCFCPUUtils
