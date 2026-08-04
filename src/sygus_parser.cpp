#include "sygus_parser.h"
#include "simba_limits.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

// A minimal S-expression: either an atom, or a list of S-expressions.
// No attempt is made to understand SMT-LIB/SyGuS grammar beyond this -
// primitive/constant detection and example extraction are done by pattern
// matching over this generic tree (see parse_sygus_bv_file's doc comment).
struct SExpr {
  bool is_atom = true;
  std::string atom;
  std::vector<SExpr> list;
};

std::string strip_comments(const std::string &text) {
  std::string out;
  out.reserve(text.size());
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    auto pos = line.find(';');
    out.append(line, 0, pos);
    out.push_back('\n');
  }
  return out;
}

std::vector<std::string> tokenize(const std::string &text) {
  std::vector<std::string> tokens;
  std::string current;
  auto flush = [&]() {
    if (!current.empty()) {
      tokens.push_back(current);
      current.clear();
    }
  };
  for (char c : text) {
    if (c == '(' || c == ')') {
      flush();
      tokens.emplace_back(1, c);
    } else if (std::isspace(static_cast<unsigned char>(c))) {
      flush();
    } else {
      current.push_back(c);
    }
  }
  flush();
  return tokens;
}

size_t parse_expr(const std::vector<std::string> &tokens, size_t pos,
                  SExpr &out) {
  if (pos >= tokens.size())
    throw std::string("Error: unexpected end of SyGuS file while parsing.");

  if (tokens[pos] == "(") {
    out.is_atom = false;
    ++pos;
    while (pos < tokens.size() && tokens[pos] != ")") {
      SExpr child;
      pos = parse_expr(tokens, pos, child);
      out.list.push_back(std::move(child));
    }
    if (pos >= tokens.size())
      throw std::string("Error: unbalanced parentheses in SyGuS file.");
    ++pos; // consume ')'
    return pos;
  }

  out.is_atom = true;
  out.atom = tokens[pos];
  return pos + 1;
}

std::vector<SExpr> parse_top_level(const std::string &text) {
  auto tokens = tokenize(strip_comments(text));
  std::vector<SExpr> forms;
  size_t pos = 0;
  while (pos < tokens.size()) {
    SExpr form;
    pos = parse_expr(tokens, pos, form);
    forms.push_back(std::move(form));
  }
  return forms;
}

bool is_hex_literal(const std::string &atom) {
  return atom.size() > 2 && atom[0] == '#' && atom[1] == 'x';
}

// Decodes a #xHEX literal for the given SyGuS bit width, truncating 64-bit
// values to their low 32 bits (with a one-time warning) since SIMBA is
// uint32_t-only. 8- and 32-bit values fit directly.
uint32_t decode_hex_literal(const std::string &atom, int bit_width,
                            bool &warned_truncation) {
  uint64_t value = std::stoull(atom.substr(2), nullptr, 16);

  if (bit_width == 64) {
    if (!warned_truncation) {
      std::cerr
          << "Warning: benchmark uses (BitVec 64); SIMBA operates on "
             "32-bit words. Examples/constants have been truncated to "
             "their lower 32 bits. Any formula found is verified correct "
             "against the truncated examples, but is not guaranteed "
             "equivalent to the original 64-bit specification (exact for "
             "bitwise/add/sub/mul, not for shift/div/rem).\n";
      warned_truncation = true;
    }
    return static_cast<uint32_t>(value & 0xFFFFFFFFull);
  }

  if (bit_width == 8 || bit_width == 32)
    return static_cast<uint32_t>(value);

  throw std::string("Error: unsupported (BitVec " + std::to_string(bit_width) +
                    "); SIMBA only supports 8-, 32- and 64-bit benchmarks.");
}

// Maps a bvXXX primitive name to the OpFlags group it belongs to. Names not
// recognised (ite, =, comparisons, ...) are silently ignored: they're either
// out of scope (ite - see the DSL extension plan) or don't gate any op.
void apply_primitive(const std::string &atom, OpFlags &flags) {
  if (atom == "bvnot" || atom == "bvand" || atom == "bvor" || atom == "bvxor")
    flags.bitwise = true;
  else if (atom == "bvneg" || atom == "bvadd" || atom == "bvsub" ||
          atom == "bvmul")
    flags.arithmetic = true;
  else if (atom == "bvshl" || atom == "bvlshr" || atom == "bvashr")
    flags.shift = true;
  else if (atom == "bvudiv" || atom == "bvurem" || atom == "bvsdiv" ||
          atom == "bvsrem")
    flags.division = true;
}

void add_constant(std::vector<uint32_t> &constants, uint32_t value) {
  if (static_cast<int>(constants.size()) >= max_n_constants)
    return; // Cap reached; keep only the first max_n_constants seen.
  for (uint32_t existing : constants)
    if (existing == value)
      return;
  constants.push_back(value);
}

// Recursively walks a grammar (or a macro body it references), collecting
// which primitive groups are used and which constants appear. define-fun
// macros (e.g. the "ehad"/"im" helpers PBE_BV_Track benchmarks define) are
// inlined by name so the bvXXX primitives and constants inside them are
// found too.
void walk_grammar(const SExpr &expr,
                  const std::unordered_map<std::string, const SExpr *> &macros,
                  int bit_width, OpFlags &flags,
                  std::vector<uint32_t> &constants, bool &warned_truncation) {
  if (expr.is_atom) {
    if (is_hex_literal(expr.atom)) {
      add_constant(constants,
                  decode_hex_literal(expr.atom, bit_width, warned_truncation));
      return;
    }
    apply_primitive(expr.atom, flags);
    auto macro = macros.find(expr.atom);
    if (macro != macros.end())
      walk_grammar(*macro->second, macros, bit_width, flags, constants,
                  warned_truncation);
    return;
  }
  for (const SExpr &child : expr.list)
    walk_grammar(child, macros, bit_width, flags, constants,
                warned_truncation);
}

} // namespace

SygusParseResult parse_sygus_bv_file(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::string("Error: could not open file " + path);

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::vector<SExpr> forms = parse_top_level(buffer.str());

  // Index define-fun macros by name, and find the synth-fun form.
  std::unordered_map<std::string, const SExpr *> macros;
  const SExpr *synth_fun = nullptr;
  for (const SExpr &form : forms) {
    if (form.is_atom || form.list.empty())
      continue;
    const std::string &head = form.list[0].atom;
    if (head == "define-fun" && form.list.size() >= 5) {
      macros[form.list[1].atom] = &form.list[4];
    } else if (head == "synth-fun" && form.list.size() >= 5) {
      synth_fun = &form;
    }
  }

  if (synth_fun == nullptr)
    throw std::string("Error: no synth-fun found in " + path);

  const std::string &fun_name = synth_fun->list[1].atom;
  const SExpr &params = synth_fun->list[2];
  const SExpr &ret_type = synth_fun->list[3];
  const SExpr &grammar = synth_fun->list[4];

  int n_variables = static_cast<int>(params.list.size());
  if (n_variables > max_n_variables)
    throw std::string("Error: " + path + " declares " +
                      std::to_string(n_variables) +
                      " variables, this version supports at most " +
                      std::to_string(max_n_variables));

  if (ret_type.is_atom || ret_type.list.size() < 2 ||
      ret_type.list[0].atom != "BitVec")
    throw std::string("Error: could not determine bit width in " + path);
  int bit_width = std::stoi(ret_type.list[1].atom);

  SygusParseResult result;
  result.instance.n_variables = n_variables;

  bool warned_truncation = false;
  walk_grammar(grammar, macros, bit_width, result.flags, result.constants,
              warned_truncation);

  // Extract literal examples: (constraint (= (fun_name <hex...>) <hex>)).
  for (const SExpr &form : forms) {
    if (form.is_atom || form.list.size() != 2 ||
        form.list[0].atom != "constraint")
      continue;
    const SExpr &eq = form.list[1];
    if (eq.is_atom || eq.list.size() != 3 || eq.list[0].atom != "=")
      continue;

    const SExpr &call = eq.list[1];
    const SExpr &output = eq.list[2];
    if (call.is_atom || call.list.empty() || call.list[0].atom != fun_name)
      continue;
    if (static_cast<int>(call.list.size()) - 1 != n_variables)
      continue;
    if (!output.is_atom || !is_hex_literal(output.atom))
      continue;

    std::vector<uint32_t> inputs;
    bool all_literals = true;
    for (size_t i = 1; i < call.list.size(); ++i) {
      if (!call.list[i].is_atom || !is_hex_literal(call.list[i].atom)) {
        all_literals = false;
        break;
      }
      inputs.push_back(
          decode_hex_literal(call.list[i].atom, bit_width, warned_truncation));
    }
    if (!all_literals)
      continue;

    for (uint32_t v : inputs)
      result.instance.inputs.push_back(v);
    result.instance.outputs.push_back(
        decode_hex_literal(output.atom, bit_width, warned_truncation));
  }

  result.instance.n_examples = static_cast<int>(result.instance.outputs.size());
  if (result.instance.n_examples == 0)
    throw std::string(
        "Error: no literal (constraint (= (" + fun_name +
        " ...) ...)) examples found in " + path +
        " - this parser only supports PBE-style SyGuS-BV files with "
        "concrete examples, not specification-style files.");

  return result;
}
