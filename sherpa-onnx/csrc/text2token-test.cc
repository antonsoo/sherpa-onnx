// sherpa-onnx/csrc/text2token-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_onnx {

// Please refer to
// https://github.com/pkufool/sherpa-test-data
// to download test data for testing
static const char dir[] = "/tmp/sherpa-test-data";

TEST(TEXT2TOKEN, TEST_cjkchar) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_cn.txt";

  std::string tokens = oss.str();

  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_cjkchar()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);

  std::string text =
      "世界人民大团结\n中国 V S 美国\n\n";  // Test blank lines also

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r = EncodeHotwords(iss, "cjkchar", sym_table, nullptr, &ids, &scores);

  std::vector<std::vector<int32_t>> expected_ids(
      {{379, 380, 72, 874, 93, 1251, 489}, {262, 147, 3423, 2476, 21, 147}});
  EXPECT_EQ(ids, expected_ids);

  EXPECT_EQ(scores.size(), 0);
}

TEST(TEXT2TOKEN, TEST_bpe) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_en.txt";
  std::string tokens = oss.str();
  oss.clear();
  oss.str("");
  oss << dir << "/text2token/bpe_en.vocab";
  std::string bpe = oss.str();
  if (!FileExists(tokens) || !FileExists(bpe)) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_bpe()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);
  auto bpe_processor = std::make_unique<ssentencepiece::Ssentencepiece>(bpe);

  std::string text = "HELLO WORLD\nI LOVE YOU :2.0";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r =
      EncodeHotwords(iss, "bpe", sym_table, bpe_processor.get(), &ids, &scores);

  std::vector<std::vector<int32_t>> expected_ids(
      {{22, 58, 24, 425}, {19, 370, 47}});
  EXPECT_EQ(ids, expected_ids);

  std::vector<float> expected_scores({0, 2.0});
  EXPECT_EQ(scores, expected_scores);
}

TEST(TEXT2TOKEN, TEST_cjkchar_bpe) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_mix.txt";
  std::string tokens = oss.str();
  oss.clear();
  oss.str("");
  oss << dir << "/text2token/bpe_mix.vocab";
  std::string bpe = oss.str();
  if (!FileExists(tokens) || !FileExists(bpe)) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_cjkchar_bpe()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);
  auto bpe_processor = std::make_unique<ssentencepiece::Ssentencepiece>(bpe);

  std::string text = "世界人民 GOES TOGETHER :1.5\n中国 GOES WITH 美国 :0.5";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r = EncodeHotwords(iss, "cjkchar+bpe", sym_table, bpe_processor.get(),
                          &ids, &scores);

  std::vector<std::vector<int32_t>> expected_ids(
      {{1368, 1392, 557, 680, 275, 178, 475},
       {685, 736, 275, 178, 179, 921, 736}});
  EXPECT_EQ(ids, expected_ids);

  std::vector<float> expected_scores({1.5, 0.5});
  EXPECT_EQ(scores, expected_scores);
}

TEST(TEXT2TOKEN, TEST_bbpe) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_bbpe.txt";
  std::string tokens = oss.str();
  oss.clear();
  oss.str("");
  oss << dir << "/text2token/bbpe.vocab";
  std::string bpe = oss.str();
  if (!FileExists(tokens) || !FileExists(bpe)) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_bbpe()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);
  auto bpe_processor = std::make_unique<ssentencepiece::Ssentencepiece>(bpe);

  std::string text = "频繁 :1.0\n李鞑靼";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r =
      EncodeHotwords(iss, "bpe", sym_table, bpe_processor.get(), &ids, &scores);

  std::vector<std::vector<int32_t>> expected_ids(
      {{259, 1118, 234, 188, 132}, {259, 1585, 236, 161, 148, 236, 160, 191}});
  EXPECT_EQ(ids, expected_ids);

  std::vector<float> expected_scores({1.0, 0});
  EXPECT_EQ(scores, expected_scores);
}

TEST(TEXT2TOKEN, TEST_invalid_score_or_threshold) {
  SymbolTable sym_table("H 1\nE 2\nL 3\nO 4\n", false);

  for (const char *line :
       {"H E L L O #", "H E L L O :high", "H E L L O :1e99"}) {
    std::istringstream iss(line);
    std::vector<std::vector<int32_t>> ids;
    std::vector<std::string> keywords;
    std::vector<float> scores;
    std::vector<float> thresholds;
    EXPECT_FALSE(
        EncodeKeywords(iss, sym_table, &ids, &keywords, &scores, &thresholds))
        << line;
  }

  std::istringstream iss("H E L L O :2.5 #0.25");
  std::vector<std::vector<int32_t>> ids;
  std::vector<std::string> keywords;
  std::vector<float> scores;
  std::vector<float> thresholds;
  EXPECT_TRUE(
      EncodeKeywords(iss, sym_table, &ids, &keywords, &scores, &thresholds));
  EXPECT_EQ(ids, (std::vector<std::vector<int32_t>>{{1, 2, 3, 3, 4}}));
  EXPECT_EQ(scores, std::vector<float>{2.5});
  EXPECT_EQ(thresholds, std::vector<float>{0.25});
}

TEST(TEXT2TOKEN, TEST_hotwords_invalid_score_or_no_tokens) {
  SymbolTable sym_table("\xe4\xbd\xa0 1\n\xe5\xa5\xbd 2\n", false);  // 你 好

  // An empty score, a line with only an invalid UTF-8 byte, a valid line
  std::istringstream iss(
      "\xe4\xbd\xa0\xe5\xa5\xbd :\n"
      "\x80\n"
      "\xe4\xbd\xa0\xe5\xa5\xbd :2.0\n");
  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;
  EXPECT_FALSE(
      EncodeHotwords(iss, "cjkchar", sym_table, nullptr, &ids, &scores));
  EXPECT_EQ(ids, (std::vector<std::vector<int32_t>>{{1, 2}, {1, 2}}));
  EXPECT_EQ(scores, (std::vector<float>{0, 2.0}));
}

}  // namespace sherpa_onnx
