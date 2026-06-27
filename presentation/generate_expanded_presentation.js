const fs = require("fs");
const path = require("path");
const PptxGenJS = require("/Users/lichenghuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT = __dirname;
const PPTX_PATH = path.join(OUT, "汇报PPT.pptx");
const NOTES_MD = path.join(OUT, "演讲稿.md");
const LEGACY_NOTES_MD = path.join(OUT, "speaker_notes_7min.md");
const CHART = path.join(ROOT, "results", "final_metrics_chart.png");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "李乘黄、马啸、THAM WAN HEI、化润宇";
pptx.subject = "社交媒体情感分析的跨域泛化";
pptx.title = "社交媒体情感分析的跨域泛化：扩展汇报版";
pptx.company = "数据挖掘课程项目";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "PingFang SC",
  bodyFontFace: "PingFang SC",
  lang: "zh-CN",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const C = {
  navy: "0F172A",
  slate: "334155",
  muted: "64748B",
  line: "CBD5E1",
  faint: "F8FAFC",
  blue: "2563EB",
  blue2: "DBEAFE",
  green: "16A34A",
  green2: "DCFCE7",
  orange: "F59E0B",
  orange2: "FEF3C7",
  red: "DC2626",
  red2: "FEE2E2",
  white: "FFFFFF",
};
const FONT = "PingFang SC";

const slideNotes = [];

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: FONT,
    margin: 0.04,
    breakLine: false,
    fit: "shrink",
    color: C.navy,
    ...opts,
  });
}

function addFooter(slide, num, source = "Source: repository results and final report") {
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55, y: 7.03, w: 12.2, h: 0,
    line: { color: "E2E8F0", width: 0.8 },
  });
  addText(slide, source, 0.62, 7.08, 8.6, 0.22, { fontSize: 6.8, color: C.muted });
  addText(slide, String(num).padStart(2, "0"), 12.25, 7.05, 0.45, 0.24, {
    fontSize: 8.5, bold: true, color: C.muted, align: "right",
  });
}

function title(slide, eyebrow, main, sub, num, source) {
  addText(slide, eyebrow, 0.62, 0.33, 2.2, 0.24, {
    fontSize: 8.3, bold: true, color: C.blue, charSpace: 1.2,
  });
  addText(slide, main, 0.62, 0.62, 8.9, 0.58, {
    fontSize: 23, bold: true, color: C.navy,
  });
  if (sub) addText(slide, sub, 0.64, 1.18, 9.6, 0.42, { fontSize: 10.8, color: C.slate });
  addFooter(slide, num, source);
}

function card(slide, x, y, w, h, head, body, color = C.blue, fill = C.blue2) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.06,
    fill: { color: fill },
    line: { color, width: 1 },
  });
  addText(slide, head, x + 0.14, y + 0.12, w - 0.28, 0.25, {
    fontSize: 8.8, bold: true, color,
  });
  addText(slide, body, x + 0.14, y + 0.42, w - 0.28, h - 0.52, {
    fontSize: 9.6, color: C.navy, valign: "mid",
    breakLine: false,
  });
}

function metric(slide, x, y, label, value, note, color = C.blue) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w: 2.15, h: 1.25,
    rectRadius: 0.06,
    fill: { color: C.white },
    line: { color: "D8E0EA", width: 1 },
    shadow: { type: "outer", color: "D9DEE8", opacity: 0.16, blur: 1, angle: 45, distance: 1 },
  });
  addText(slide, value, x + 0.12, y + 0.12, 1.9, 0.42, { fontSize: 22, bold: true, color });
  addText(slide, label, x + 0.13, y + 0.57, 1.9, 0.24, { fontSize: 8.6, bold: true, color: C.navy });
  addText(slide, note, x + 0.13, y + 0.86, 1.9, 0.25, { fontSize: 7.2, color: C.muted });
}

function table(slide, rows, x, y, colW, rowH, opts = {}) {
  const data = rows.map((r, i) => r.map((cell) => ({
    text: String(cell),
    options: {
      fontFace: FONT,
      fontSize: opts.fontSize || 8.2,
      color: i === 0 ? C.white : C.navy,
      bold: i === 0,
      fill: { color: i === 0 ? C.slate : (i % 2 === 0 ? C.faint : C.white) },
      margin: 0.06,
      valign: "mid",
    },
  })));
  slide.addTable(data, {
    x, y, w: colW.reduce((a, b) => a + b, 0),
    h: rowH * rows.length,
    colW,
    rowH,
    border: { color: C.line, width: 0.55 },
  });
}

function bullets(slide, items, x, y, w, h, opts = {}) {
  const text = items.map((it) => `• ${it}`).join("\n");
  addText(slide, text, x, y, w, h, {
    fontSize: opts.fontSize || 11,
    color: opts.color || C.navy,
    breakLine: false,
    fit: "shrink",
    valign: "top",
    paraSpaceAfterPt: 8,
  });
}

function sectionTag(slide, x, y, label, color = C.blue) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w: 1.58, h: 0.32,
    rectRadius: 0.04,
    fill: { color },
    line: { color },
  });
  addText(slide, label, x + 0.08, y + 0.08, 1.42, 0.14, {
    fontSize: 6.8, bold: true, color: C.white, align: "center",
  });
}

function addSlide(num, eyebrow, main, sub, note, source) {
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  title(slide, eyebrow, main, sub, num, source);
  slide.addNotes(note);
  slideNotes.push({ num, title: main, note });
  return slide;
}

function addMiniBar(slide, x, y, label, value, max, color) {
  addText(slide, label, x, y, 1.45, 0.22, { fontSize: 8, color: C.slate });
  slide.addShape(pptx.ShapeType.rect, { x: x + 1.55, y: y + 0.04, w: 2.7, h: 0.12, fill: { color: "E2E8F0" }, line: { color: "E2E8F0" } });
  slide.addShape(pptx.ShapeType.rect, { x: x + 1.55, y: y + 0.04, w: 2.7 * value / max, h: 0.12, fill: { color }, line: { color } });
  addText(slide, value.toFixed(3), x + 4.35, y, 0.6, 0.22, { fontSize: 8, bold: true, color });
}

// 1
{
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  addText(slide, "社交媒体情感分析的跨域泛化", 0.72, 0.58, 8.0, 0.62, { fontSize: 29, bold: true, color: C.navy });
  addText(slide, "面向分布偏移的 Data-Centric 改进与可复现实验系统", 0.74, 1.22, 7.8, 0.36, { fontSize: 14, color: C.slate });
  sectionTag(slide, 0.78, 1.78, "FINAL DEFENSE", C.blue);
  addText(slide, "数据挖掘课程项目 · 扩展汇报版", 0.78, 2.18, 4.3, 0.24, { fontSize: 10, color: C.muted });
  addText(slide, "小组成员\n李乘黄 3220251475\n马啸 3220251484\nTHAM WAN HEI 3820251057\n化润宇 3220251231", 0.78, 4.72, 4.5, 1.15, { fontSize: 10.2, color: C.navy, breakLine: false });
  addText(slide, "日期：2026-06-13", 0.78, 6.08, 3.2, 0.26, { fontSize: 9.5, color: C.muted });
  metric(slide, 9.1, 0.78, "目标域微博", "6000", "含 3600 条 unlabeled", C.blue);
  metric(slide, 9.1, 2.25, "最佳目标 F1", "0.927", "E3 target adapter", C.green);
  metric(slide, 9.1, 3.72, "最平衡 DeltaF1", "-0.019", "E6 char1-5 backbone", C.orange);
  card(slide, 7.05, 5.42, 5.15, 0.92, "一句话结论", "少量目标域标注 + 弱监督/适配策略，能显著缩小评论源域到微博目标域的情感分类性能衰减。", C.blue, "EFF6FF");
  addFooter(slide, 1, "Artifacts: final_report, final_metrics, final_data_audit");
  const note = "大家好，我们小组的项目是《社交媒体情感分析的跨域泛化》。这次扩展版汇报仍然围绕一个核心问题：在评论数据上训练好的模型，为什么迁移到微博这种社交媒体文本后会明显失效，以及我们怎样用数据层改进和可复现实验去缩小这个差距。封面右侧三个数字先给出结论：目标域有 6000 条微博样本，其中 3600 条用于弱监督伪标注；最佳目标域 Macro-F1 达到 0.927；最平衡的跨域衰减 DeltaF1 是 -0.019。后面的汇报会依次说明问题、数据、方法、证据和最终交付。";
  slide.addNotes(note);
  slideNotes.push({ num: 1, title: "封面", note });
}

// 2
{
  const slide = addSlide(2, "AGENDA", "汇报顺序保持五段结构", "问题 → 数据 → 方法 → 证据 → 演示/总结", "汇报结构保持原来的五部分，但每一部分增加一页细节或证据。第一部分讲问题背景和评价约束；第二部分讲数据来源、划分和偏移审计；第三部分讲系统架构、方法矩阵和训练评估协议；第四部分讲实验结果、消融、负向发现和失败案例；第五部分总结贡献、局限和演示过渡。这样页数从 10 页扩展到 20 页，但叙事顺序不变。", "Source: repository results and final report");
  const items = [
    ["01", "问题背景", "为什么跨域情感分析值得做"],
    ["02", "数据介绍", "来源、规模、偏移与审计"],
    ["03", "技术方案", "架构、基线、进阶模块"],
    ["04", "实验结果", "指标、消融、失败案例"],
    ["05", "总结演示", "贡献、局限与系统 demo"],
  ];
  items.forEach((it, i) => {
    const y = 1.92 + i * 0.74;
    addText(slide, it[0], 1.05, y, 0.55, 0.28, { fontSize: 14, bold: true, color: C.blue });
    addText(slide, it[1], 1.75, y - 0.01, 1.55, 0.28, { fontSize: 13, bold: true, color: C.navy });
    addText(slide, it[2], 3.35, y + 0.02, 4.4, 0.24, { fontSize: 9.6, color: C.slate });
    slide.addShape(pptx.ShapeType.line, { x: 1.0, y: y + 0.45, w: 6.9, h: 0, line: { color: "E2E8F0", width: 0.7 } });
  });
  card(slide, 8.62, 2.02, 3.25, 2.18, "扩充原则", "不改变章节顺序；每个原主题拆成主张页和证据页；新增页只补足解释、协议、风险和演示材料。", C.green, C.green2);
  card(slide, 8.62, 4.48, 3.25, 1.05, "时间建议", "20 页约 13-14 分钟；可按需要删减证据页。", C.orange, C.orange2);
}

// 3
{
  const slide = addSlide(3, "MOTIVATION", "问题背景：源域模型直接迁移到微博会遇到双重偏移", "文本风格偏移 + 类别分布偏移共同导致目标域性能衰减", "这一页解释为什么这个题目不是简单的情感分类。源域是外卖评论，表达方式更集中在口味、配送、服务等消费体验上，标签也比较均衡。目标域是微博，文本里经常有转发链、表情、反讽和社会事件语境，而且我们刻意构造成负:正=2:1，模拟舆情监测中负面声音偏多的场景。因此，源域学到的词面模式迁移到微博时会失效，负面召回也会变得更重要。", "Source: final_data_audit.json");
  card(slide, 0.9, 1.95, 3.35, 2.25, "源域：评论文本", "waimai_10k 外卖评论\n短文本、商品/服务体验\n标签均衡，平均 22.9 字\n容易学到口味、配送、服务词", C.blue, C.blue2);
  card(slide, 4.95, 1.95, 3.35, 2.25, "目标域：微博文本", "weibo_senti_100k 微博情感\n转发链、表情、反讽\n负面偏重，平均 51.9 字\n更接近舆情监测", C.green, C.green2);
  card(slide, 9.0, 1.95, 3.05, 2.25, "Domain shift", "同一情感标签在两个域里的词面、语境和类别比例都不同；目标域不能只看 accuracy。", C.orange, C.orange2);
  bullets(slide, ["业务价值：更早发现负面舆情，负面召回率必须足够高", "科学价值：量化 F1(source) 与 F1(target) 的衰减", "工程价值：形成数据、实验、报告一键复现闭环"], 1.02, 4.75, 10.7, 1.05, { fontSize: 11.2 });
}

// 4
{
  const slide = addSlide(4, "PROBLEM DETAIL", "评价目标不能只看目标域 F1", "DeltaF1、负面召回和 AUC 共同约束跨域泛化", "这一页补充评价目标。跨域任务里，单看目标域 F1 不够，因为模型可能通过牺牲源域稳定性换取目标域分数，也可能因为阈值选择导致负面召回不足。所以我们同时看 Macro-F1、Weighted-F1、AUC、DeltaF1 和目标域负面召回率。DeltaF1 越接近 0，说明源域和目标域表现越平衡；负面召回率则对应舆情监测中漏报负面信息的风险。", "Source: README and final_metrics.json");
  metric(slide, 0.95, 1.85, "跨域衰减", "DeltaF1", "F1(source) - F1(target)", C.blue);
  metric(slide, 3.48, 1.85, "舆情风险", "Neg R", "目标域负面召回", C.green);
  metric(slide, 6.0, 1.85, "阈值无关", "AUC", "排序能力", C.orange);
  metric(slide, 8.53, 1.85, "主指标", "Macro-F1", "类别不均衡更稳健", C.red);
  addText(slide, "验收逻辑", 1.05, 3.85, 1.5, 0.3, { fontSize: 15, bold: true });
  bullets(slide, ["如果 DeltaF1 很大，说明跨域衰减仍然明显", "如果负面召回低，舆情场景会漏掉关键负面评论", "如果只优化 MMD/KL，可能降低分布距离但损伤情感触发词", "最终实验必须同时报告均值和标准差，避免单个随机种子偶然性"], 1.05, 4.35, 10.9, 1.6, { fontSize: 11.2 });
}

// 5
{
  const slide = addSlide(5, "DATA AUDIT", "数据介绍：正式数据不是示例数据", "源域 8000 条，目标域 6000 条，目标域满足 5k+ 要求", "这页说明数据规模。源域来自 ChineseNlpCorpus 的 waimai_10k，处理后 8000 条；目标域来自 HuggingFace 的 weibo_senti_100k，处理后 6000 条。目标域进一步切分成 train、val、test 和 unlabeled，其中 3600 条未标注样本用于弱监督伪标注。这一点很关键，因为中期阶段可能只是 smoke benchmark，而最终版已经换成真实公开语料，并且目标域满足 5k+ 的规模要求。", "Source: results/final_data_audit.json");
  table(slide, [
    ["域", "raw rows", "processed", "train", "val", "test", "unlabeled", "正例比例", "平均长度"],
    ["源域 waimai", "11987", "8000", "5600", "1200", "1200", "0", "50.0%", "22.9"],
    ["目标域 weibo", "119000", "6000", "600", "600", "1200", "3600", "33.3%", "51.9"],
  ], 0.7, 1.82, [1.45, 1.0, 1.05, 0.8, 0.75, 0.75, 1.1, 0.9, 0.85], 0.46, { fontSize: 7.8 });
  metric(slide, 1.0, 4.15, "目标域负:正", "2:1", "模拟负面舆情偏重", C.orange);
  metric(slide, 3.55, 4.15, "未标注样本", "3600", "用于弱监督伪标注", C.blue);
  metric(slide, 6.1, 4.15, "目标域金标训练", "10%", "少量标注适配", C.green);
  card(slide, 8.7, 4.05, 3.25, 1.35, "审计产物", "results/final_data_audit.json\n记录原始行数、采样规模、类别比例、平均长度和下载来源。", C.slate, C.faint);
}

// 6
{
  const slide = addSlide(6, "DATA SPLIT", "数据划分为方法设计服务", "目标域少量金标 + 大量 unlabeled 支撑 adapter 和 pseudo labels", "这一页把数据划分和方法联系起来。源域主要用于训练基础情感分类能力；目标域 train 很小，只占 600 条，用于模拟现实中人工标注昂贵的情况；目标域 unlabeled 有 3600 条，适合弱监督伪标注；目标域 val 用来调阈值，尤其要约束负面召回；test 则只在最终报告中使用。这样做的好处是每个模块都对应一个明确的数据来源，避免训练和评估混在一起。", "Source: scripts/prepare_real_data.py");
  const boxes = [
    ["source train", "5600", "学习评论域基础情感特征", C.blue],
    ["target train", "600", "少量金标校准 adapter", C.green],
    ["target unlabeled", "3600", "高置信弱监督伪标注", C.orange],
    ["target val/test", "600 / 1200", "调阈值与最终评估分离", C.red],
  ];
  boxes.forEach((b, i) => {
    const x = 0.75 + i * 3.05;
    card(slide, x, 2.0, 2.58, 1.7, b[0], `${b[1]}\n${b[2]}`, b[3], i === 0 ? C.blue2 : i === 1 ? C.green2 : i === 2 ? C.orange2 : C.red2);
    if (i < 3) addText(slide, "→", x + 2.72, 2.58, 0.3, 0.3, { fontSize: 20, bold: true, color: C.muted });
  });
  bullets(slide, ["目标域验证集负责选择阈值 tau，避免测试集泄漏", "unlabeled 只通过高置信伪标注进入训练，不直接参与测试", "负面召回率作为阈值约束，服务于舆情业务目标"], 1.0, 4.55, 10.8, 1.15, { fontSize: 11.4 });
}

// 7
{
  const slide = addSlide(7, "DATA SHIFT", "数据偏移来自文本风格、长度和语义触发词", "微博目标域更长、更噪，且正负词面冲突更频繁", "这一页进一步解释为什么目标域更难。微博文本平均长度约 51.9 字，比源域评论长得多，里面常常有转发链、表情符号、社会事件实体和反讽表达。源域里的“好吃、配送、服务”等词，迁移到微博时不一定有同样意义；微博里的“哈哈、祝福、鼓掌”也可能出现在负面或讽刺语境中。因此我们不仅需要模型，还需要错误案例和失败模式分析。", "Source: final_error_cases.csv");
  card(slide, 0.85, 1.78, 3.1, 2.2, "长度差异", "源域平均 22.9 字\n目标域平均 51.9 字\n更容易出现跨句语义", C.blue, C.blue2);
  card(slide, 4.25, 1.78, 3.1, 2.2, "噪声差异", "转发链、表情、话题标签\n文本片段之间语义跳跃\n局部词面不可靠", C.orange, C.orange2);
  card(slide, 7.65, 1.78, 3.1, 2.2, "类别差异", "源域标签均衡\n目标域负面偏重\n评估必须关注负面召回", C.green, C.green2);
  addText(slide, "结论", 0.96, 4.55, 0.8, 0.28, { fontSize: 15, bold: true });
  addText(slide, "数据偏移不是一个单独模块能解决的问题，所以后续实验把伪标注、目标域校准、领域过滤和 backbone 替换放进同一消融矩阵。", 1.0, 4.98, 10.6, 0.62, { fontSize: 12, color: C.navy, breakLine: false });
}

// 8
{
  const slide = addSlide(8, "ARCHITECTURE", "技术方案：从公开语料到可复现实验闭环", "每一步都有文件产物，方便验收、复跑和定位问题", "这一页展示系统架构。流程从公开语料开始，经过 TextCleaner 做去噪、去重和字段统一，然后生成 source_full 和 social_full 两个正式数据文件。接着进入特征编码和 E0-E6 实验矩阵，每个方法跑 3 个随机种子，并在目标域验证集调阈值。最后输出 JSON、CSV、PNG 图表和 PDF/Markdown 报告。架构重点不是只跑一次模型，而是让数据、代码、结果和报告可以互相追溯。", "Flow source: scripts/prepare_real_data.py, src/experiments/final_runner.py");
  const steps = [
    ["公开语料", "waimai_10k\nweibo_senti_100k"],
    ["清洗去噪", "TextCleaner\n去重、过滤噪声"],
    ["数据切分", "train / val / test\nunlabeled 目标域"],
    ["特征编码", "char n-gram\n情感词典特征"],
    ["实验矩阵", "E0-E6\n3 seeds"],
    ["阈值评估", "Macro-F1 / AUC\nNeg Recall"],
    ["结果报告", "metrics / summary\nerror cases / report"],
  ];
  steps.forEach((s, i) => {
    const x = 0.55 + i * 1.78;
    card(slide, x, 2.05, 1.42, 1.18, s[0], s[1], i < 4 ? C.blue : C.green, i < 4 ? C.blue2 : C.green2);
    if (i < steps.length - 1) addText(slide, "→", x + 1.48, 2.48, 0.25, 0.25, { fontSize: 16, bold: true, color: C.muted });
  });
  bullets(slide, ["`verify_final.py` 检查数据规模、E0-E6 完整性、每个方法 3 seeds 和达标指标", "`final_metrics.json` 与 `final_summary.csv` 是报告数字来源", "`final_error_cases.csv` 支撑失败案例分析"], 0.9, 4.55, 11.2, 1.2, { fontSize: 11.2 });
}

// 9
{
  const slide = addSlide(9, "REPRODUCIBILITY", "可复现性设计：不是只交 PPT，而是交实验系统", "运行脚本、验证脚本、指标文件和报告生成链路一一对应", "这一页补充可复现性。最终交付不只是 PPT 和报告，而是包含数据准备、实验运行、报告生成和验证脚本。别人可以运行 run_final 重新生成结果，也可以用 verify_final 快速检查已有产物是否完整。报告中的数据规模、实验指标和错误样本都来自本地结果文件，而不是手工填写。这个设计能减少答辩时最常见的问题：数字从哪里来、结果能不能复跑、失败案例是不是临时编的。", "Source: README, scripts/verify_final.py");
  table(slide, [
    ["阶段", "命令/文件", "作用"],
    ["数据准备", "scripts/prepare_real_data.py", "下载并生成 source/social full split"],
    ["实验运行", "./scripts/run_final.sh", "执行 E0-E6、3 seeds"],
    ["报告生成", "scripts/generate_final_report.py", "读取 metrics/audit 生成报告"],
    ["交付验收", "scripts/verify_final.py", "检查数据、指标、报告和达标条件"],
  ], 0.95, 1.85, [1.35, 3.45, 5.3], 0.52, { fontSize: 8.4 });
  card(slide, 1.05, 5.0, 4.65, 0.88, "复现命令", "./scripts/run_final.sh\npython scripts/verify_final.py", C.blue, C.blue2);
  card(slide, 6.15, 5.0, 4.75, 0.88, "快速验收", "检查 target >= 5000、E0-E6、3 seeds、DeltaF1 与负面召回", C.green, C.green2);
}

// 10
{
  const slide = addSlide(10, "METHOD", "技术方案：E0-E6 隔离每个模块的贡献", "同一数据、同一指标、三随机种子，避免单模型故事", "这一页是方法矩阵。E0 是 TF-IDF+LR 的源域基线，给出下限；E1 增强 lexical encoder，加入字符 n-gram 和情感词典；E2 加入弱监督伪标注；E3 使用少量目标域金标做 adapter calibration；E4 组合伪标注和 adapter；E5 做领域特征过滤；E6 使用 char1-5 更宽的特征空间作为 backbone 替换。这样每个模块都有可比较的对照。", "Source: src/experiments/final_runner.py");
  table(slide, [
    ["ID", "方法", "关键变化"],
    ["E0", "TF-IDF+LR", "仅源域训练，浅层下限"],
    ["E1", "strong lexical", "char n-gram + 情感词典"],
    ["E2", "pseudo labels", "加入高置信目标域伪标注"],
    ["E3", "target adapter", "少量目标域金标校准"],
    ["E4", "pseudo + adapter", "组合 E2 与 E3"],
    ["E5", "feature filter", "过滤强领域特异 n-gram"],
    ["E6", "char1-5 backbone", "扩大特征空间"],
  ], 0.85, 1.65, [0.72, 2.35, 4.7], 0.43, { fontSize: 8.2 });
  card(slide, 8.45, 1.75, 3.1, 2.2, "设计原则", "每次只改变一个主要模块；报告均值±标准差；同时观察 F1、AUC、DeltaF1 和负面召回。", C.blue, C.blue2);
  card(slide, 8.45, 4.28, 3.1, 1.1, "核心对照", "E1→E2 看伪标注\nE1→E3 看校准\nE4→E5 看领域过滤", C.orange, C.orange2);
}

// 11
{
  const slide = addSlide(11, "TRAINING PROTOCOL", "训练与评估协议：阈值选择也纳入实验", "目标域验证集调 tau，并约束负面召回不低于业务要求", "这一页解释训练逻辑。对于每个方法，训练数据从源域开始，如果启用伪标注，就加入目标域 unlabeled 中的高置信样本；如果启用 adapter，就加入加权的目标域 train 金标样本。模型训练后，不直接用默认 0.5 阈值，而是在目标域验证集上搜索最优阈值，同时让负面召回尽量不低于 0.85。最后再在源域和目标域测试集上报告结果。", "Source: src/experiments/final_runner.py");
  addText(slide, "核心逻辑伪代码", 0.9, 1.72, 2.2, 0.28, { fontSize: 14, bold: true });
  slide.addShape(pptx.ShapeType.roundRect, { x: 0.9, y: 2.15, w: 5.35, h: 2.35, rectRadius: 0.06, fill: { color: "F1F5F9" }, line: { color: "CBD5E1" } });
  addText(slide, "for method in E0..E6:\n  train_data = source\n  if pseudo: add high_conf(unlabeled)\n  if adapter: add weighted(target_train)\n  model = sparse_logreg(features)\n  tau = argmax F1(target_val)\n        with neg_recall >= 0.85\n  report source_test / target_test", 1.15, 2.35, 4.85, 1.92, {
    fontFace: "Menlo",
    fontSize: 10.2,
    color: C.navy,
    breakLine: false,
  });
  card(slide, 6.95, 2.05, 2.1, 1.45, "为什么调阈值", "类别不均衡下默认阈值可能漏掉负面评论。", C.green, C.green2);
  card(slide, 9.35, 2.05, 2.1, 1.45, "为什么 3 seeds", "降低单次采样、模型初始化和伪标注选择的偶然性。", C.orange, C.orange2);
  bullets(slide, ["源域测试集衡量基础能力是否被破坏", "目标域测试集衡量迁移效果", "DeltaF1 衡量跨域性能差距是否真正收窄"], 6.95, 4.05, 4.7, 1.1, { fontSize: 10.7 });
}

// 12
{
  const slide = addSlide(12, "BASELINE GAP", "基线证明：source-only 迁移存在明显衰减", "E0/E1 在目标域 F1 只有 0.625/0.644，不能满足最终目标", "这一页先看基线。E0 的目标域 F1 只有 0.625，E1 增强 lexical 特征后也只有 0.644，说明单纯把源域模型迁移到微博仍然有明显衰减。E1 的负面召回达到 0.873，但正例召回较弱，说明它更偏向预测负面，不能代表整体分类能力好。因此后面需要伪标注、目标域校准和更宽特征空间来补充目标域信号。", "Source: results/final_summary.csv");
  metric(slide, 1.05, 1.9, "E0 Target F1", "0.625", "source-only 浅层基线", C.red);
  metric(slide, 3.55, 1.9, "E1 Target F1", "0.644", "增强 lexical 后仍不足", C.orange);
  metric(slide, 6.05, 1.9, "E1 DeltaF1", "0.222", "跨域差距明显", C.blue);
  metric(slide, 8.55, 1.9, "E1 Neg Recall", "0.873", "偏负面但整体 F1 低", C.green);
  addText(slide, "基线结论", 1.05, 4.0, 1.2, 0.3, { fontSize: 15, bold: true });
  bullets(slide, ["增强特征本身不能解决微博语境和标签偏移", "负面召回高不等于模型全面好，因为正例召回会被牺牲", "后续模块必须同时提升目标域 F1 并控制 DeltaF1"], 1.05, 4.48, 10.5, 1.2, { fontSize: 11.2 });
}

// 13
{
  const slide = addSlide(13, "RESULTS", "实验结果：E3 目标域 F1 最佳，E6 跨域衰减最平衡", "Target Macro-F1 / Negative Recall / Target AUC", "这一页展示最终结果图。蓝色是目标域 Macro-F1，绿色是负面召回，黄色是目标域 AUC。可以看到 E3 的目标域 F1 最高，达到 0.927，负面召回也达到 0.958；E6 的目标域 F1 是 0.890，但 DeltaF1 只有 -0.019，是源域和目标域表现最平衡的方法。E5 虽然负面召回不低，但 F1 明显回落，后面会作为负向消融分析。", "Source: results/final_summary.csv");
  if (fs.existsSync(CHART)) {
    slide.addImage({ path: CHART, x: 0.9, y: 1.55, w: 7.1, h: 4.02 });
  } else {
    addText(slide, "final_metrics_chart.png missing", 1.0, 2.4, 4.0, 0.3, { fontSize: 13, color: C.red });
  }
  metric(slide, 8.75, 1.75, "E3 Target F1", "0.927", "少量目标域金标校准最有效", C.green);
  metric(slide, 8.75, 3.25, "E6 DeltaF1", "-0.019", "源域/目标域性能最均衡", C.orange);
  metric(slide, 8.75, 4.75, "E3 Neg Recall", "0.958", "满足舆情场景 Recall 要求", C.blue);
}

// 14
{
  const slide = addSlide(14, "RESULT INTERPRETATION", "为什么 E3 与 E6 是两个不同意义上的最优", "一个追求目标域最高分，一个追求跨域稳定性", "这一页解释结果，不只是报数字。E3 使用少量目标域金标样本做校准，所以它最直接地补上了目标域语境和类别分布信息，目标域 F1 最好。E6 使用 char1-5 更宽特征空间，它未必达到最高目标 F1，但源域和目标域之间的差距最小，说明特征覆盖更均衡。因此如果业务只关心微博舆情上线效果，可以优先考虑 E3；如果课程实验强调跨域泛化稳定性，E6 是很重要的补充结论。", "Source: results/final_metrics.json");
  card(slide, 0.95, 1.85, 4.45, 2.15, "E3：目标域性能最优", "Target F1=0.927\nNeg Recall=0.958\n原因：少量目标域金标直接校准微博语境与阈值。", C.green, C.green2);
  card(slide, 6.1, 1.85, 4.45, 2.15, "E6：跨域衰减最平衡", "Target F1=0.890\nDeltaF1=-0.019\n原因：char1-5 扩大特征覆盖，源域和目标域表现更接近。", C.orange, C.orange2);
  addText(slide, "汇报时的落点", 1.0, 4.55, 1.8, 0.28, { fontSize: 14.5, bold: true });
  bullets(slide, ["最高目标域效果和最佳泛化平衡不是同一个问题", "E3 适合实际部署时优先追求目标域效果", "E6 证明 backbone/特征空间会影响跨域稳定性"], 1.0, 4.95, 10.4, 1.08, { fontSize: 11.2 });
}

// 15
{
  const slide = addSlide(15, "ABLATION", "消融证明：伪标注和校准有效", "E1→E2→E3 展示目标域信号逐步增强", "这一页讲正向消融。E1 是增强特征但只用源域训练，目标 F1 为 0.644。加入弱监督伪标注后，E2 提升到 0.753，说明 unlabeled 目标域数据确实提供了有用的目标域词面信号。E3 加入少量目标域金标校准后提升到 0.927，是最大提升来源。这个结果说明，在跨域情感分析里，少量高质量目标域标注比盲目堆模型更关键。", "Source: results/final_summary.csv");
  addMiniBar(slide, 1.1, 2.0, "E1 source-only", 0.644, 1.0, C.blue);
  addMiniBar(slide, 1.1, 2.55, "E2 + pseudo", 0.753, 1.0, C.green);
  addMiniBar(slide, 1.1, 3.1, "E3 + adapter", 0.927, 1.0, C.orange);
  card(slide, 7.1, 1.85, 3.95, 1.18, "E2 的贡献", "无标注目标域样本经高置信筛选后，补充微博词面和表达习惯。", C.green, C.green2);
  card(slide, 7.1, 3.32, 3.95, 1.18, "E3 的贡献", "少量目标域金标样本直接修正类别边界和阈值，是收益最大模块。", C.orange, C.orange2);
  addText(slide, "目标 F1 提升路径：0.644 → 0.753 → 0.927", 1.1, 4.95, 7.2, 0.38, { fontSize: 18, bold: true, color: C.navy });
}

// 16
{
  const slide = addSlide(16, "NEGATIVE ABLATION", "负向发现：分布距离降低不等于分类效果更好", "E5 降低 MMD，但目标 F1 只有 0.681", "这一页讲负向消融。E5 做领域特征过滤，目标是去掉强领域特异 n-gram，模拟 domain-adversarial 的效果。它确实把 MMD 从 0.049 降到 0.016，但目标 F1 只有 0.681，明显低于 E2、E3 和 E6。原因是过滤过强时，可能把微博目标域里的关键情感触发词也删掉了。这说明跨域对齐不能只看分布距离，还要看任务指标和错误样本。", "Source: results/final_summary.csv");
  metric(slide, 1.0, 1.85, "MMD", "0.016", "分布距离降低", C.green);
  metric(slide, 3.55, 1.85, "E5 Target F1", "0.681", "分类效果回落", C.red);
  metric(slide, 6.1, 1.85, "E5 Neg Recall", "0.898", "负面召回尚可", C.orange);
  card(slide, 8.65, 1.78, 3.0, 1.95, "解释", "强行删除领域特异 n-gram 会降低域差异，但也可能删除目标域关键情感词。", C.red, C.red2);
  addText(slide, "方法失效案例的意义", 1.05, 4.35, 2.4, 0.3, { fontSize: 14.5, bold: true });
  bullets(slide, ["如果只报告成功方法，会掩盖方法边界", "E5 说明数据对齐指标必须和分类指标一起看", "后续可以改为软权重过滤，而不是硬删除特征"], 1.05, 4.8, 10.6, 1.0, { fontSize: 11.2 });
}

// 17
{
  const slide = addSlide(17, "FAILURE CASE", "失败案例：正向词并不总是正向情绪", "微博文本中局部情感词与整体语境经常冲突", "这一页展示具体错误。样本里有很多“美味、太香、不错”这样的正向词，所以 E0 预测为正面；但真实标签是负面，因为整体语境里有“吃多了、抓狂”等负面体验。线性 n-gram 模型容易抓住局部词面，却没有能力理解跨句语义和反讽。这类失败案例说明，最终模型即使指标提升，也仍然需要上下文编码器、表情特征和困难样本主动学习来继续改进。", "Source: results/final_error_cases.csv");
  slide.addShape(pptx.ShapeType.roundRect, { x: 0.85, y: 1.62, w: 7.05, h: 2.2, rectRadius: 0.08, fill: { color: "F8FAFC" }, line: { color: C.line } });
  addText(slide, "真实错误样本（E0）", 1.08, 1.86, 2.2, 0.28, { fontSize: 12.5, bold: true, color: C.navy });
  addText(slide, "“今天又吃多了[抓狂]这劲爆的音乐配上如此的美味，实在无法抗拒啊！黑菌意大利米饭配烤奶酪片太香……现在已经飘飘然了！”", 1.08, 2.27, 6.45, 0.85, { fontSize: 11, color: C.navy, breakLine: false });
  addText(slide, "真实标签：负面 / 预测：正面 / positive prob=0.619", 1.08, 3.24, 4.7, 0.24, { fontSize: 9.2, color: C.red, bold: true });
  card(slide, 8.35, 1.62, 3.25, 1.18, "为什么会错？", "局部有正向词；整体语境却包含负面体验和反讽。", C.orange, C.orange2);
  card(slide, 8.35, 3.02, 3.25, 1.18, "改进方向", "加入上下文编码器、表情/反讽特征，并建立困难样本集。", C.green, C.green2);
  bullets(slide, ["错误类型用于指导下一轮主动学习采样", "失败案例证明 lexical/hash 特征仍有边界"], 1.0, 4.65, 10.5, 0.72, { fontSize: 11.2 });
}

// 18
{
  const slide = addSlide(18, "LIMITATIONS", "局限性：CPU 友好版本牺牲了深层语义建模", "最终实验可复现，但不是语义能力的上限", "这一页承认局限。为了保证课程环境中无 GPU、无外部账号也能复现，最终可运行实验使用 lexical/hash 特征来模拟完整迁移框架。这让 E0-E6 消融非常快、非常稳定，但对长转发链、反讽、表情和跨句语义的建模能力弱于 BERT/RoBERTa。后续工作可以在保持同一数据 split 和同一评估协议的基础上，加入上下文编码器、人工一致性评估和更精细的错误类型标注。", "Source: final_report.md");
  card(slide, 0.9, 1.85, 3.25, 1.75, "当前局限", "lexical/hash 特征速度快、可复现，但缺少深层上下文语义。", C.red, C.red2);
  card(slide, 4.55, 1.85, 3.25, 1.75, "风险场景", "反讽、表情组合、长转发链、社会事件实体可能造成误判。", C.orange, C.orange2);
  card(slide, 8.2, 1.85, 3.25, 1.75, "后续升级", "接入 BERT/RoBERTa 微调，补人工一致性和主动学习。", C.green, C.green2);
  addText(slide, "保持可比性的前提", 0.98, 4.38, 2.2, 0.28, { fontSize: 14.5, bold: true });
  bullets(slide, ["不更换数据 split，保证和当前 E0-E6 结果可比", "沿用 Macro-F1、AUC、DeltaF1、Neg Recall 指标", "把 final_error_cases.csv 作为困难样本评估集的一部分"], 1.0, 4.83, 10.6, 1.0, { fontSize: 11.2 });
}

// 19
{
  const slide = addSlide(19, "DELIVERABLES", "最终交付：数据、实验、报告和答辩材料形成闭环", "每个关键结论都有文件证据支撑", "这一页总结交付物。项目不仅有 PPT，还有 processed 数据、final_metrics、final_summary、final_error_cases、final_report 和演示脚本。verify_final 可以检查数据规模、目标域 unlabeled、E0-E6 是否完整、每个方法是否 3 个 seed，以及是否有方法满足 DeltaF1 和负面召回要求。答辩时如果被问到结果来源，可以直接指向这些文件。", "Source: repository deliverables");
  table(slide, [
    ["类别", "文件", "证明内容"],
    ["数据", "data/processed/source_full.csv", "源域 8000 条"],
    ["数据", "data/processed/social_full.csv", "目标域 6000 条，含 3600 unlabeled"],
    ["指标", "results/final_metrics.json", "E0-E6 全量运行结果"],
    ["摘要", "results/final_summary.csv", "均值、标准差和关键指标"],
    ["错误", "results/final_error_cases.csv", "失败案例与分析依据"],
    ["报告", "report/final_report.pdf", "最终文字报告"],
  ], 0.75, 1.55, [1.0, 4.0, 4.9], 0.42, { fontSize: 8.2 });
  card(slide, 1.0, 5.25, 4.9, 0.78, "验收命令", "python scripts/verify_final.py", C.blue, C.blue2);
  card(slide, 6.25, 5.25, 4.9, 0.78, "演示衔接", "3 分钟系统 demo 展示文件和验证脚本", C.green, C.green2);
}

// 20
{
  const slide = addSlide(20, "WRAP-UP", "总结：目标域校准最有效，跨域平衡需要更宽特征空间", "扩展版汇报结束，接 3 分钟系统演示", "最后总结三点贡献。第一，使用真实公开数据构建了源域 8000 条、目标域 6000 条的跨域情感任务；第二，完成 E0-E6 消融矩阵，每个方法 3 个随机种子并报告均值和标准差；第三，形成了数据准备、实验运行、结果验证和报告生成的可复现闭环。关键结论是：E3 目标域校准最有效，E6 跨域衰减最平衡，E5 提醒我们不能只看分布距离。接下来可以进入 3 分钟系统演示，展示这些产物如何被验证。", "Source: repository results and final report");
  metric(slide, 0.95, 1.85, "最佳目标 F1", "0.927", "E3 target adapter", C.green);
  metric(slide, 3.48, 1.85, "最平衡 DeltaF1", "-0.019", "E6 char1-5 backbone", C.orange);
  metric(slide, 6.0, 1.85, "目标域规模", "6000", "真实公开微博数据", C.blue);
  card(slide, 8.55, 1.8, 3.05, 1.55, "一句话收束", "跨域情感分析的核心不是堆模型，而是识别分布偏移，并用可验证的数据策略缩小衰减。", C.slate, C.faint);
  bullets(slide, ["真实数据：源域 8000，目标域 6000，含 unlabeled", "完整消融：E0-E6、3 seeds、均值±标准差", "可复现闭环：prepare_real_data / run_final / verify_final / report"], 1.0, 4.35, 10.4, 1.1, { fontSize: 11.5 });
  addText(slide, "演示过渡：打开仓库 → 查看产物 → 运行 verify_final → 展示 summary/error cases/report", 1.0, 6.0, 10.6, 0.35, { fontSize: 11.2, bold: true, color: C.blue });
}

function buildNotesMarkdown() {
  const lines = [];
  lines.push("# 14 分钟答辩讲稿");
  lines.push("");
  lines.push("建议语速：每分钟 180-220 个中文字符。整体结构保持“问题背景 → 数据介绍 → 技术方案 → 实验结果 → 总结演示”，页数从 10 页扩展到 20 页。");
  lines.push("");
  for (const item of slideNotes) {
    lines.push(`## Slide ${item.num} ${item.title}`);
    lines.push("");
    lines.push(item.note);
    lines.push("");
  }
  return lines.join("\n");
}

(async () => {
  const md = buildNotesMarkdown();
  fs.writeFileSync(NOTES_MD, md, "utf8");
  fs.writeFileSync(LEGACY_NOTES_MD, md, "utf8");
  await pptx.writeFile({ fileName: PPTX_PATH });
  console.log(`Wrote ${PPTX_PATH}`);
  console.log(`Wrote ${NOTES_MD}`);
})();
