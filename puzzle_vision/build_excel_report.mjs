import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  FileBlob,
  SpreadsheetFile,
  Workbook,
} from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const workspace = path.dirname(scriptDir);
const finalDir = path.join(
  scriptDir,
  "batch_results_rule2_random_strict_v3",
);
const priorDir = path.join(
  scriptDir,
  "batch_results_rule2_random_strict_v2",
);
const archiveDir = path.join(
  scriptDir,
  "异常样本_规则2随机裁剪_失败或接近阈值",
);
const outputDir = path.join(
  workspace,
  "outputs/vision_regression_rule2_random_strict",
);
const outputPath = path.join(
  outputDir,
  "视觉回归测试_现场随机尺寸裁剪_52张牌_严格版.xlsx",
);
const previewDir = path.join(outputDir, "previews");

const cases = JSON.parse(await fs.readFile(path.join(finalDir, "cases.json"), "utf8"));
const pieces = JSON.parse(
  await fs.readFile(path.join(finalDir, "piece_errors.json"), "utf8"),
);
const seams = JSON.parse(
  await fs.readFile(path.join(finalDir, "seam_errors.json"), "utf8"),
);
const finalSummary = JSON.parse(
  await fs.readFile(path.join(finalDir, "summary.json"), "utf8"),
);
const priorCases = JSON.parse(
  (await fs.readFile(path.join(priorDir, "cases.json"), "utf8")).replace(
    /\bNaN\b/g,
    "null",
  ),
);
const priorSummary = JSON.parse(
  await fs.readFile(path.join(priorDir, "summary.json"), "utf8"),
);
const archiveSummary = JSON.parse(
  await fs.readFile(path.join(archiveDir, "异常样本汇总.json"), "utf8"),
);

const thresholdRows = [
  ["source_center_max_mm", "单组最大源中心定位误差", 1.5, "mm", "视觉定位预算，给机械控制保留余量"],
  ["source_angle_max_deg", "单组最大源角度误差", 2.0, "deg", "电磁铁独立旋转角控制的视觉预算"],
  ["source_vertex_chamfer_max_mm", "单组最大顶点 Chamfer 误差", 2.0, "mm", "约束拼图片形状拟合"],
  ["source_area_error_max_percent", "单组最大单片面积误差", 5.0, "%", "检测轮廓相对真值面积误差"],
  ["nominal_long_error_mm", "重构长边误差", 3.0, "mm", "相对该组随机目标长边真值"],
  ["nominal_short_error_mm", "重构短边误差", 3.0, "mm", "相对该组随机目标短边真值"],
  ["relative_layout_rms_mm", "去除预留间距后的布局 RMS", 2.0, "mm", "衡量整体拼图相对结构"],
  ["planned_vertex_gap_max_mm", "计划放置最大顶点间距", 8.0, "mm", "内部严格值；规则上限为 20 mm"],
  ["rigid_edge_error_max_mm", "刚体变换最大边长变化", 0.00001, "mm", "验证 P2 等碎片不变形"],
  ["rigid_area_error_max_mm2", "刚体变换最大面积变化", 0.00001, "mm²", "验证移动只含平移和旋转"],
];
const thresholdRowByMetric = Object.fromEntries(
  thresholdRows.map((row, index) => [row[0], index + 5]),
);
const thresholdValueByMetric = Object.fromEntries(
  thresholdRows.map((row) => [row[0], row[2]]),
);
const chartLabels = [
  "中心定位",
  "角度定位",
  "顶点拟合",
  "面积拟合",
  "重构长边",
  "重构短边",
  "布局RMS",
  "顶点间距",
  "刚体边长",
  "刚体面积",
];

function colLetter(oneBased) {
  let value = oneBased;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function metricRatio(caseRow, metric) {
  if (caseRow[metric] === null || caseRow[metric] === undefined) {
    return Number.NaN;
  }
  const value = Number(caseRow[metric]);
  const threshold = thresholdValueByMetric[metric];
  return Number.isFinite(value) ? value / threshold : Number.NaN;
}

function nearMetrics(caseRow, ratio = 0.8) {
  return thresholdRows
    .filter(([metric]) => {
      const value = metricRatio(caseRow, metric);
      return Number.isFinite(value) && value >= ratio;
    })
    .map(([metric]) => metric);
}

function maximumFiniteMetricRatio(caseRow) {
  const ratios = thresholdRows
    .map(([metric]) => metricRatio(caseRow, metric))
    .filter(Number.isFinite);
  return ratios.length > 0 ? Math.max(...ratios) : null;
}

function numericOrBlank(value) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  return value ?? null;
}

const colors = {
  navy: "#16324F",
  blue: "#2563EB",
  blueLight: "#DBEAFE",
  green: "#16856B",
  greenLight: "#DDF4EC",
  yellow: "#FFF2CC",
  orange: "#F59E0B",
  red: "#C2413A",
  redLight: "#FDE2E2",
  gray50: "#F8FAFC",
  gray100: "#F1F5F9",
  gray300: "#CBD5E1",
  gray600: "#475569",
  white: "#FFFFFF",
};

function styleTitle(sheet, rangeAddress, title, subtitle) {
  const titleRange = sheet.getRange(rangeAddress);
  titleRange.merge();
  titleRange.values = [[title]];
  titleRange.format.fill = colors.navy;
  titleRange.format.font = {
    bold: true,
    color: colors.white,
    size: 18,
  };
  titleRange.format.verticalAlignment = "center";
  titleRange.format.rowHeight = 30;
  const startCol = rangeAddress.split(":")[0].replace(/[0-9]/g, "");
  const endCol = rangeAddress.split(":")[1].replace(/[0-9]/g, "");
  const subtitleRange = sheet.getRange(`${startCol}2:${endCol}2`);
  subtitleRange.merge();
  subtitleRange.values = [[subtitle]];
  subtitleRange.format.fill = colors.gray100;
  subtitleRange.format.font = { color: colors.gray600, size: 10 };
  subtitleRange.format.wrapText = true;
  subtitleRange.format.rowHeight = 28;
}

function styleHeader(range) {
  range.format.fill = colors.blue;
  range.format.font = { bold: true, color: colors.white };
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.wrapText = true;
  range.format.rowHeight = 30;
  range.format.borders = {
    preset: "all",
    style: "thin",
    color: colors.gray300,
  };
}

function styleBody(range) {
  range.format.borders = {
    preset: "all",
    style: "thin",
    color: colors.gray300,
  };
  range.format.verticalAlignment = "center";
}

function addTable(sheet, startRow, endRow, endColumn, name) {
  const table = sheet.tables.add(
    `A${startRow}:${endColumn}${endRow}`,
    true,
    name,
  );
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  return table;
}

const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("数据概览");
const thresholds = workbook.worksheets.add("严格阈值");
const caseSheet = workbook.worksheets.add("组级结果");
const pieceSheet = workbook.worksheets.add("单片误差");
const seamSheet = workbook.worksheets.add("拼接边误差");
const cardSheet = workbook.worksheets.add("52牌面汇总");
const screenshotSheet = workbook.worksheets.add("截图索引");
const anomalySheet = workbook.worksheets.add("异常样本");

// Strict thresholds
thresholds.showGridLines = false;
styleTitle(
  thresholds,
  "A1:F1",
  "严格阈值与判定依据",
  "黄色单元格为可编辑阈值；组级结果中的 Excel 判定和阈值占用率均引用本表。",
);
thresholds.getRange("A4:F4").values = [[
  "指标键",
  "中文说明",
  "严格阈值",
  "单位",
  "用途",
  "图表标签",
]];
styleHeader(thresholds.getRange("A4:F4"));
thresholds.getRange("A5:F14").values = thresholdRows.map((row, index) => [
  ...row,
  chartLabels[index],
]);
thresholds.getRange("C5:C14").format.fill = colors.yellow;
thresholds.getRange("C5:C14").format.font = { bold: true, color: colors.navy };
thresholds.getRange("A5:F14").format.borders = {
  preset: "all",
  style: "thin",
  color: colors.gray300,
};
thresholds.getRange("C5:C12").format.numberFormat = "0.000";
thresholds.getRange("C13:C14").format.numberFormat = "0.000000";
thresholds.getRange("A:F").format.autofitColumns();
thresholds.getRange("A:A").format.columnWidth = 31;
thresholds.getRange("B:B").format.columnWidth = 27;
thresholds.getRange("E:E").format.columnWidth = 38;
thresholds.getRange("F:F").format.columnWidth = 14;
thresholds.freezePanes.freezeRows(1);

// Case-level results
const caseFields = [
  "case_id",
  "card",
  "rank",
  "suit",
  "layout_index",
  "layout_seed",
  "target_seed",
  "expected_piece_count",
  "target_width_mm",
  "target_height_mm",
  "elapsed_ms",
  "solver_score",
  "source_center_max_mm",
  "source_center_mean_mm",
  "source_angle_max_deg",
  "source_angle_mean_deg",
  "source_vertex_chamfer_max_mm",
  "source_vertex_chamfer_mean_mm",
  "source_area_error_max_percent",
  "source_area_error_mean_percent",
  "nominal_long_mm",
  "nominal_short_mm",
  "target_long_mm",
  "target_short_mm",
  "nominal_long_error_mm",
  "nominal_short_error_mm",
  "relative_layout_rms_mm",
  "planned_vertex_gap_max_mm",
  "planned_vertex_gap_mean_mm",
  "rigid_edge_error_max_mm",
  "rigid_area_error_max_mm2",
  "passed_python",
  "failed_checks_python",
  "mode",
  "piece_count",
  "input_image",
  "segmentation_image",
  "detection_image",
  "solution_image",
  "reconstructed_image",
  "contact_sheet",
  "error",
];
const caseHeaders = [
  "测试组ID",
  "牌面",
  "点数",
  "花色",
  "布局序号",
  "随机种子",
  "裁剪种子",
  "期望碎片数",
  "目标宽度(mm)",
  "目标高度(mm)",
  "耗时(ms)",
  "求解评分",
  "中心误差最大(mm)",
  "中心误差均值(mm)",
  "角度误差最大(°)",
  "角度误差均值(°)",
  "顶点误差最大(mm)",
  "顶点误差均值(mm)",
  "面积误差最大(%)",
  "面积误差均值(%)",
  "重构长边(mm)",
  "重构短边(mm)",
  "目标长边(mm)",
  "目标短边(mm)",
  "长边误差(mm)",
  "短边误差(mm)",
  "相对布局RMS(mm)",
  "计划顶点间距最大(mm)",
  "计划顶点间距均值(mm)",
  "刚体边长变化最大(mm)",
  "刚体面积变化最大(mm²)",
  "Python判定",
  "Python失败项",
  "识别模式",
  "碎片数",
  "输入图路径",
  "分割图路径",
  "检测图路径",
  "解算图路径",
  "重建图路径",
  "汇总图路径",
  "异常信息",
  "阈值占用率",
  "Excel判定",
  "异常分类",
];
const caseStartRow = 5;
const caseEndRow = caseStartRow + cases.length - 1;
const occupancyColumn = caseFields.length + 1;
const excelResultColumn = caseFields.length + 2;
const anomalyColumn = caseFields.length + 3;
const occupancyLetter = colLetter(occupancyColumn);
const excelResultLetter = colLetter(excelResultColumn);
const anomalyLetter = colLetter(anomalyColumn);
const caseLastLetter = anomalyLetter;
const caseFieldColumn = Object.fromEntries(
  caseFields.map((field, index) => [field, colLetter(index + 1)]),
);
caseSheet.showGridLines = false;
styleTitle(
  caseSheet,
  `A1:${caseLastLetter}1`,
  `${cases.length} 组现场随机几何视觉回归结果`,
  `52 张牌 × 每张 ${finalSummary.layouts_per_card} 组；每组随机目标尺寸、2–4 片合法裁剪、位置和旋转。Excel 判定引用“严格阈值”，80% 以上标为 NEAR。`,
);
caseSheet.getRange(`A4:${caseLastLetter}4`).values = [caseHeaders];
styleHeader(caseSheet.getRange(`A4:${caseLastLetter}4`));
const caseValues = cases.map((row) =>
  caseFields.map((field) => numericOrBlank(row[field])),
);
caseSheet.getRange(`A${caseStartRow}:${colLetter(caseFields.length)}${caseEndRow}`).values =
  caseValues;
const caseFormulas = cases.map((_, index) => {
  const row = caseStartRow + index;
  const ratios = thresholdRows.map(([metric]) => {
    const metricCol = caseFieldColumn[metric];
    const thresholdRow = thresholdRowByMetric[metric];
    return `${metricCol}${row}/'严格阈值'!$C$${thresholdRow}`;
  });
  const conditions = thresholdRows.map(([metric]) => {
    const metricCol = caseFieldColumn[metric];
    const thresholdRow = thresholdRowByMetric[metric];
    return `${metricCol}${row}<='严格阈值'!$C$${thresholdRow}`;
  });
  return [
    `=MAX(${ratios.join(",")})`,
    `=IF(AND(${conditions.join(",")}),"PASS","FAIL")`,
    `=IF(${excelResultLetter}${row}="FAIL","FAIL",IF(${occupancyLetter}${row}>=0.8,"NEAR","NORMAL"))`,
  ];
});
caseSheet.getRange(
  `${occupancyLetter}${caseStartRow}:${anomalyLetter}${caseEndRow}`,
).formulas = caseFormulas;
styleBody(caseSheet.getRange(`A${caseStartRow}:${caseLastLetter}${caseEndRow}`));
caseSheet.getRange(
  `${caseFieldColumn.target_width_mm}${caseStartRow}:${caseFieldColumn.rigid_area_error_max_mm2}${caseEndRow}`,
).format.numberFormat = "0.000";
caseSheet.getRange(`${occupancyLetter}${caseStartRow}:${occupancyLetter}${caseEndRow}`)
  .format.numberFormat = "0.0%";
caseSheet.getRange(`${excelResultLetter}${caseStartRow}:${anomalyLetter}${caseEndRow}`)
  .format.horizontalAlignment = "center";
caseSheet
  .getRange(`${excelResultLetter}${caseStartRow}:${excelResultLetter}${caseEndRow}`)
  .conditionalFormats.add("cellIs", {
    operator: "equal",
    formula: '"FAIL"',
    format: { fill: colors.redLight, font: { color: colors.red, bold: true } },
  });
caseSheet
  .getRange(`${anomalyLetter}${caseStartRow}:${anomalyLetter}${caseEndRow}`)
  .conditionalFormats.add("cellIs", {
    operator: "equal",
    formula: '"NEAR"',
    format: { fill: colors.yellow, font: { color: "#92400E", bold: true } },
  });
addTable(caseSheet, 4, caseEndRow, caseLastLetter, "CaseResultsTable");
caseSheet.freezePanes.freezeRows(4);
caseSheet.freezePanes.freezeColumns(2);
caseSheet.getRange(`A:${caseLastLetter}`).format.autofitColumns();
caseSheet.getRange("A:A").format.columnWidth = 35;
caseSheet.getRange("B:B").format.columnWidth = 20;
caseSheet.getRange(
  `${caseFieldColumn.input_image}:${caseFieldColumn.error}`,
).format.columnWidth = 32;

// Per-piece errors (4 rows per case)
const pieceFields = [
  "case_id",
  "card",
  "layout_index",
  "layout_seed",
  "target_seed",
  "expected_piece_count",
  "target_width_mm",
  "target_height_mm",
  "detected_piece_id",
  "original_piece_id",
  "source_x_truth_mm",
  "source_y_truth_mm",
  "source_x_detected_mm",
  "source_y_detected_mm",
  "source_center_error_mm",
  "source_angle_truth_deg",
  "source_angle_detected_deg",
  "source_angle_error_deg",
  "source_vertex_chamfer_mm",
  "source_area_truth_mm2",
  "source_area_detected_mm2",
  "source_area_error_mm2",
  "source_area_error_percent",
  "rigid_edge_error_mm",
  "rigid_area_error_mm2",
  "rotation_delta_deg",
  "target_x_mm",
  "target_y_mm",
];
const pieceHeaders = [
  "测试组ID",
  "牌面",
  "布局序号",
  "随机种子",
  "裁剪种子",
  "期望碎片数",
  "目标宽度(mm)",
  "目标高度(mm)",
  "检测片ID",
  "原始片ID",
  "真值X(mm)",
  "真值Y(mm)",
  "检测X(mm)",
  "检测Y(mm)",
  "中心误差(mm)",
  "真值角度(°)",
  "检测角度(°)",
  "角度误差(°)",
  "顶点误差(mm)",
  "真值面积(mm²)",
  "检测面积(mm²)",
  "面积误差(mm²)",
  "面积误差(%)",
  "刚体边长变化(mm)",
  "刚体面积变化(mm²)",
  "执行旋转量(°)",
  "目标X(mm)",
  "目标Y(mm)",
  "单片判定",
];
const pieceStartRow = 5;
const pieceEndRow = pieceStartRow + pieces.length - 1;
const pieceResultColumn = pieceFields.length + 1;
const pieceResultLetter = colLetter(pieceResultColumn);
const pieceFieldColumn = Object.fromEntries(
  pieceFields.map((field, index) => [field, colLetter(index + 1)]),
);
pieceSheet.showGridLines = false;
styleTitle(
  pieceSheet,
  `A1:${pieceResultLetter}1`,
  `单片误差明细（${pieces.length.toLocaleString("en-US")} 条）`,
  "每组含 2–4 块随机合法裁剪片；中心、角度、顶点、面积与刚体不变性误差均逐片记录。",
);
pieceSheet.getRange(`A4:${pieceResultLetter}4`).values = [pieceHeaders];
styleHeader(pieceSheet.getRange(`A4:${pieceResultLetter}4`));
pieceSheet.getRange(
  `A${pieceStartRow}:${colLetter(pieceFields.length)}${pieceEndRow}`,
).values = pieces.map((row) =>
  pieceFields.map((field) => numericOrBlank(row[field])),
);
pieceSheet.getRange(
  `${pieceResultLetter}${pieceStartRow}:${pieceResultLetter}${pieceEndRow}`,
).formulas = pieces.map((_, index) => {
  const row = pieceStartRow + index;
  return [
    `=IF(AND(${pieceFieldColumn.source_center_error_mm}${row}<='严格阈值'!$C$5,${pieceFieldColumn.source_angle_error_deg}${row}<='严格阈值'!$C$6,${pieceFieldColumn.source_vertex_chamfer_mm}${row}<='严格阈值'!$C$7,${pieceFieldColumn.source_area_error_percent}${row}<='严格阈值'!$C$8,${pieceFieldColumn.rigid_edge_error_mm}${row}<='严格阈值'!$C$13,${pieceFieldColumn.rigid_area_error_mm2}${row}<='严格阈值'!$C$14),"PASS","FAIL")`,
  ];
});
styleBody(pieceSheet.getRange(`A${pieceStartRow}:${pieceResultLetter}${pieceEndRow}`));
pieceSheet.getRange(
  `${pieceFieldColumn.target_width_mm}${pieceStartRow}:${pieceFieldColumn.target_y_mm}${pieceEndRow}`,
).format.numberFormat = "0.000";
addTable(pieceSheet, 4, pieceEndRow, pieceResultLetter, "PieceErrorsTable");
pieceSheet.freezePanes.freezeRows(4);
pieceSheet.freezePanes.freezeColumns(2);
pieceSheet.getRange(`A:${pieceResultLetter}`).format.autofitColumns();
pieceSheet.getRange("A:A").format.columnWidth = 35;

// Per-seam errors (3 rows per case)
const seamFields = [
  "case_id",
  "card",
  "layout_index",
  "layout_seed",
  "target_seed",
  "expected_piece_count",
  "target_width_mm",
  "target_height_mm",
  "seam_index",
  "piece_a",
  "edge_a",
  "piece_b",
  "edge_b",
  "edge_length_normalized_error",
  "endpoint_gap_1_mm",
  "endpoint_gap_2_mm",
  "seam_max_vertex_gap_mm",
];
const seamHeaders = [
  "测试组ID",
  "牌面",
  "布局序号",
  "随机种子",
  "裁剪种子",
  "期望碎片数",
  "目标宽度(mm)",
  "目标高度(mm)",
  "拼接边序号",
  "片A",
  "边A",
  "片B",
  "边B",
  "边长归一化误差",
  "端点间距1(mm)",
  "端点间距2(mm)",
  "该边最大端点间距(mm)",
  "拼接边判定",
];
const seamStartRow = 5;
const seamEndRow = seamStartRow + seams.length - 1;
const seamResultColumn = seamFields.length + 1;
const seamResultLetter = colLetter(seamResultColumn);
const seamFieldColumn = Object.fromEntries(
  seamFields.map((field, index) => [field, colLetter(index + 1)]),
);
seamSheet.showGridLines = false;
styleTitle(
  seamSheet,
  `A1:${seamResultLetter}1`,
  `拼接边误差明细（${seams.length.toLocaleString("en-US")} 条）`,
  "2/3/4 片分别形成 1/2/3 条连接关系；两个对应端点的间距均单独记录。",
);
seamSheet.getRange(`A4:${seamResultLetter}4`).values = [seamHeaders];
styleHeader(seamSheet.getRange(`A4:${seamResultLetter}4`));
seamSheet.getRange(
  `A${seamStartRow}:${colLetter(seamFields.length)}${seamEndRow}`,
).values = seams.map((row) =>
  seamFields.map((field) => numericOrBlank(row[field])),
);
seamSheet.getRange(
  `${seamResultLetter}${seamStartRow}:${seamResultLetter}${seamEndRow}`,
).formulas = seams.map((_, index) => {
  const row = seamStartRow + index;
  return [
    `=IF(${seamFieldColumn.seam_max_vertex_gap_mm}${row}<='严格阈值'!$C$12,"PASS","FAIL")`,
  ];
});
styleBody(seamSheet.getRange(`A${seamStartRow}:${seamResultLetter}${seamEndRow}`));
seamSheet.getRange(
  `${seamFieldColumn.target_width_mm}${seamStartRow}:${seamFieldColumn.seam_max_vertex_gap_mm}${seamEndRow}`,
).format.numberFormat = "0.0000";
addTable(seamSheet, 4, seamEndRow, seamResultLetter, "SeamErrorsTable");
seamSheet.freezePanes.freezeRows(4);
seamSheet.freezePanes.freezeColumns(2);
seamSheet.getRange(`A:${seamResultLetter}`).format.autofitColumns();
seamSheet.getRange("A:A").format.columnWidth = 35;

// Per-card summary, formula-backed against case-level data
const cardNames = [...new Set(cases.map((row) => row.card))].sort();
const cardStartRow = 5;
const cardEndRow = cardStartRow + cardNames.length - 1;
const cardHeaders = [
  "牌面",
  "测试组数",
  "通过组数",
  "通过率",
  "中心误差最大(mm)",
  "角度误差最大(°)",
  "顶点误差最大(mm)",
  "面积误差最大(%)",
  "长边误差最大(mm)",
  "短边误差最大(mm)",
  "布局RMS最大(mm)",
  "顶点间距最大(mm)",
  "最大耗时(ms)",
  "牌面判定",
];
const cardLastLetter = colLetter(cardHeaders.length);
cardSheet.showGridLines = false;
styleTitle(
  cardSheet,
  `A1:${cardLastLetter}1`,
  "52 张牌面覆盖统计",
  `每张牌面均包含 ${finalSummary.layouts_per_card} 组随机目标尺寸、裁剪和初始姿态；统计公式引用“组级结果”。`,
);
cardSheet.getRange(`A4:${cardLastLetter}4`).values = [cardHeaders];
styleHeader(cardSheet.getRange(`A4:${cardLastLetter}4`));
cardSheet.getRange(`A${cardStartRow}:A${cardEndRow}`).values = cardNames.map((name) => [
  name,
]);
const cardFormulaRows = cardNames.map((_, index) => {
  const row = cardStartRow + index;
  const caseCard = `'组级结果'!$${caseFieldColumn.card}$${caseStartRow}:$${caseFieldColumn.card}$${caseEndRow}`;
  return [
    `=COUNTIF(${caseCard},A${row})`,
    `=COUNTIFS(${caseCard},A${row},'组级结果'!$${excelResultLetter}$${caseStartRow}:$${excelResultLetter}$${caseEndRow},"PASS")`,
    `=IF(B${row}=0,0,C${row}/B${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.source_center_max_mm}$${caseStartRow}:$${caseFieldColumn.source_center_max_mm}$${caseEndRow},${caseCard},A${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.source_angle_max_deg}$${caseStartRow}:$${caseFieldColumn.source_angle_max_deg}$${caseEndRow},${caseCard},A${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.source_vertex_chamfer_max_mm}$${caseStartRow}:$${caseFieldColumn.source_vertex_chamfer_max_mm}$${caseEndRow},${caseCard},A${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.source_area_error_max_percent}$${caseStartRow}:$${caseFieldColumn.source_area_error_max_percent}$${caseEndRow},${caseCard},A${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.nominal_long_error_mm}$${caseStartRow}:$${caseFieldColumn.nominal_long_error_mm}$${caseEndRow},${caseCard},A${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.nominal_short_error_mm}$${caseStartRow}:$${caseFieldColumn.nominal_short_error_mm}$${caseEndRow},${caseCard},A${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.relative_layout_rms_mm}$${caseStartRow}:$${caseFieldColumn.relative_layout_rms_mm}$${caseEndRow},${caseCard},A${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.planned_vertex_gap_max_mm}$${caseStartRow}:$${caseFieldColumn.planned_vertex_gap_max_mm}$${caseEndRow},${caseCard},A${row})`,
    `=MAXIFS('组级结果'!$${caseFieldColumn.elapsed_ms}$${caseStartRow}:$${caseFieldColumn.elapsed_ms}$${caseEndRow},${caseCard},A${row})`,
    `=IF(C${row}=B${row},"PASS","FAIL")`,
  ];
});
cardSheet.getRange(`B${cardStartRow}:${cardLastLetter}${cardEndRow}`).formulas =
  cardFormulaRows;
styleBody(cardSheet.getRange(`A${cardStartRow}:${cardLastLetter}${cardEndRow}`));
cardSheet.getRange(`D${cardStartRow}:D${cardEndRow}`).format.numberFormat = "0.0%";
cardSheet.getRange(`E${cardStartRow}:M${cardEndRow}`).format.numberFormat = "0.000";
addTable(cardSheet, 4, cardEndRow, cardLastLetter, "CardSummaryTable");
cardSheet.freezePanes.freezeRows(4);
cardSheet.getRange(`A:${cardLastLetter}`).format.autofitColumns();
cardSheet.getRange("A:A").format.columnWidth = 20;

// Screenshot index with one embedded contact sheet for every case
screenshotSheet.showGridLines = false;
styleTitle(
  screenshotSheet,
  "A1:J1",
  `每组测试截图索引（${cases.length} 组）`,
  "嵌入图从左到右依次为：输入、分割、检测、解算、纹理重建。所有单图路径见“组级结果”。",
);
screenshotSheet.getRange("A4:J4").values = [[
  "测试组ID",
  "牌面",
  "布局",
  "碎片数",
  "目标宽(mm)",
  "目标高(mm)",
  "最终判定",
  "临界触发项",
  "汇总图路径",
  "汇总截图",
]];
styleHeader(screenshotSheet.getRange("A4:J4"));
const screenshotValues = cases.map((row) => {
  const triggers = nearMetrics(row, 0.8);
  return [
    row.case_id,
    row.card,
    row.layout_index,
    row.expected_piece_count,
    row.target_width_mm,
    row.target_height_mm,
    row.passed_python ? "PASS" : "FAIL",
    triggers.join("; "),
    row.contact_sheet,
    "",
  ];
});
const screenshotStartRow = 5;
const screenshotEndRow = screenshotStartRow + cases.length - 1;
screenshotSheet.getRange(`A${screenshotStartRow}:J${screenshotEndRow}`).values =
  screenshotValues;
styleBody(screenshotSheet.getRange(`A${screenshotStartRow}:J${screenshotEndRow}`));
screenshotSheet.getRange(`A${screenshotStartRow}:J${screenshotEndRow}`).format.rowHeightPx =
  112;
screenshotSheet.getRange("A:A").format.columnWidth = 35;
screenshotSheet.getRange("B:B").format.columnWidth = 20;
screenshotSheet.getRange("C:H").format.columnWidth = 14;
screenshotSheet.getRange(`E${screenshotStartRow}:F${screenshotEndRow}`).format.numberFormat =
  "0.00";
screenshotSheet.getRange("I:I").format.columnWidth = 55;
screenshotSheet.getRange("J:J").format.columnWidthPx = 470;
screenshotSheet.freezePanes.freezeRows(4);
for (let index = 0; index < cases.length; index += 1) {
  const imageBytes = await fs.readFile(cases[index].contact_sheet);
  screenshotSheet.images.add({
    dataUrl: `data:image/jpeg;base64,${imageBytes.toString("base64")}`,
    anchor: {
      from: { row: screenshotStartRow - 1 + index, col: 9 },
      extent: { widthPx: 455, heightPx: 105 },
    },
  });
}

// Anomaly archive index: prior failures plus final near-threshold cases
const anomalyHeaders = [
  "类别",
  "测试组ID",
  "牌面",
  "布局",
  "碎片数",
  "目标宽(mm)",
  "目标高(mm)",
  "触发指标",
  "阈值占用率最大",
  "原始截图目录",
  "归档目录",
];
const anomalyRows = [];
for (const row of priorCases.filter((item) => !item.passed_python)) {
  anomalyRows.push([
    "首轮失败_算法改进前",
    row.case_id,
    row.card,
    row.layout_index,
    row.expected_piece_count,
    row.target_width_mm,
    row.target_height_mm,
    row.failed_checks_python,
    maximumFiniteMetricRatio(row),
    path.dirname(row.contact_sheet),
    path.join(
      archiveDir,
      "首轮失败_算法改进前",
      row.card,
      `layout_${String(row.layout_index).padStart(2, "0")}`,
    ),
  ]);
}
for (const row of cases) {
  const triggers = nearMetrics(row, 0.8);
  if (!row.passed_python || triggers.length > 0) {
    const category = row.passed_python ? "最终临界通过_阈值80pct" : "最终失败";
    anomalyRows.push([
      category,
      row.case_id,
      row.card,
      row.layout_index,
      row.expected_piece_count,
      row.target_width_mm,
      row.target_height_mm,
      triggers.join("; ") || row.failed_checks_python,
      maximumFiniteMetricRatio(row),
      path.dirname(row.contact_sheet),
      path.join(
        archiveDir,
        category,
        row.card,
        `layout_${String(row.layout_index).padStart(2, "0")}`,
      ),
    ]);
  }
}
const anomalyStartRow = 5;
const anomalyEndRow = anomalyStartRow + anomalyRows.length - 1;
anomalySheet.showGridLines = false;
styleTitle(
  anomalySheet,
  "A1:K1",
  "失败或接近阈值样本",
  `判定规则：失败全收录；通过但任一指标达到阈值 80% 以上也收录。当前共 ${archiveSummary.archived_cases_total} 组。`,
);
anomalySheet.getRange("A4:K4").values = [anomalyHeaders];
styleHeader(anomalySheet.getRange("A4:K4"));
anomalySheet.getRange(`A${anomalyStartRow}:K${anomalyEndRow}`).values = anomalyRows;
styleBody(anomalySheet.getRange(`A${anomalyStartRow}:K${anomalyEndRow}`));
anomalySheet.getRange(`I${anomalyStartRow}:I${anomalyEndRow}`).format.numberFormat =
  "0.0%";
anomalySheet.getRange(`F${anomalyStartRow}:G${anomalyEndRow}`).format.numberFormat =
  "0.00";
addTable(anomalySheet, 4, anomalyEndRow, "K", "AnomalyIndexTable");
anomalySheet.getRange("A:K").format.autofitColumns();
anomalySheet.getRange("A:A").format.columnWidth = 28;
anomalySheet.getRange("B:B").format.columnWidth = 35;
anomalySheet.getRange("J:K").format.columnWidth = 55;
anomalySheet.freezePanes.freezeRows(4);

// Dashboard
dashboard.showGridLines = false;
styleTitle(
  dashboard,
  "A1:Q1",
  "E 题现场随机尺寸与裁剪视觉回归：52 张牌严格验证",
  `共 ${cases.length} 组：目标矩形尺寸、2–4 片合法裁剪、初始位置和旋转均变化；控制函数仍为安全占位。`,
);
dashboard.getRange("A4:N4").values = [[
  "总测试组",
  "",
  "严格通过",
  "",
  "通过率",
  "",
  "最终临界组",
  "",
  "最终失败",
  "",
  "单片记录",
  "",
  "拼接边记录",
  "",
]];
dashboard.getRange("A4:N4").format.fill = colors.gray100;
dashboard.getRange("A4:N4").format.font = { bold: true, color: colors.gray600 };
dashboard.getRange("A5:B6").merge();
dashboard.getRange("C5:D6").merge();
dashboard.getRange("E5:F6").merge();
dashboard.getRange("G5:H6").merge();
dashboard.getRange("I5:J6").merge();
dashboard.getRange("K5:L6").merge();
dashboard.getRange("M5:N6").merge();
dashboard.getRange("A5").formulas = [[
  `=COUNTA('组级结果'!$A$${caseStartRow}:$A$${caseEndRow})`,
]];
dashboard.getRange("C5").formulas = [[
  `=COUNTIF('组级结果'!$${excelResultLetter}$${caseStartRow}:$${excelResultLetter}$${caseEndRow},"PASS")`,
]];
dashboard.getRange("E5").formulas = [["=C5/A5"]];
dashboard.getRange("G5").formulas = [[
  `=COUNTIF('组级结果'!$${anomalyLetter}$${caseStartRow}:$${anomalyLetter}$${caseEndRow},"NEAR")`,
]];
dashboard.getRange("I5").formulas = [[
  `=COUNTIF('组级结果'!$${excelResultLetter}$${caseStartRow}:$${excelResultLetter}$${caseEndRow},"FAIL")`,
]];
dashboard.getRange("K5").formulas = [[
  `=COUNTA('单片误差'!$A$${pieceStartRow}:$A$${pieceEndRow})`,
]];
dashboard.getRange("M5").formulas = [[
  `=COUNTA('拼接边误差'!$A$${seamStartRow}:$A$${seamEndRow})`,
]];
for (const address of ["A5:B6", "C5:D6", "E5:F6", "G5:H6", "I5:J6", "K5:L6", "M5:N6"]) {
  const range = dashboard.getRange(address);
  range.format.fill = colors.greenLight;
  range.format.font = { bold: true, color: colors.green, size: 20 };
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.borders = {
    preset: "outside",
    style: "medium",
    color: colors.green,
  };
}
dashboard.getRange("E5").format.numberFormat = "0.0%";

dashboard.getRange("A8:E8").values = [[
  "严格指标",
  "最终最大值",
  "严格阈值",
  "阈值占用率",
  "结论",
]];
styleHeader(dashboard.getRange("A8:E8"));
const dashboardMetricRows = thresholdRows.map(([metric, description], index) => {
  const row = 9 + index;
  const caseCol = caseFieldColumn[metric];
  return [
    description,
    `=MAX('组级结果'!$${caseCol}$${caseStartRow}:$${caseCol}$${caseEndRow})`,
    `='严格阈值'!$C$${thresholdRowByMetric[metric]}`,
    `=B${row}/C${row}`,
    `=IF(D${row}<=1,"PASS","FAIL")`,
  ];
});
dashboard.getRange("A9:A18").values = dashboardMetricRows.map((row) => [row[0]]);
dashboard.getRange("B9:E18").formulas = dashboardMetricRows.map((row) => row.slice(1));
styleBody(dashboard.getRange("A9:E18"));
dashboard.getRange("B9:C18").format.numberFormat = "0.0000";
dashboard.getRange("D9:D18").format.numberFormat = "0.0%";
dashboard.getRange("E9:E18").format.horizontalAlignment = "center";

dashboard.getRange("G8:H8").values = [["指标", "阈值占用率"]];
styleHeader(dashboard.getRange("G8:H8"));
dashboard.getRange("G9:G18").formulas = Array.from({ length: 10 }, (_, index) => [
  `='严格阈值'!$F$${5 + index}`,
]);
dashboard.getRange("H9:H18").formulas = Array.from({ length: 10 }, (_, index) => [
  `=D${9 + index}`,
]);
dashboard.getRange("H9:H18").format.numberFormat = "0%";
const occupancyChart = dashboard.charts.add("bar", dashboard.getRange("G8:H18"));
occupancyChart.setPosition("J8", "Q25");
occupancyChart.title = "最终最大误差 / 严格阈值";
occupancyChart.hasLegend = false;
occupancyChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
occupancyChart.yAxis = { numberFormatCode: "0%", min: 0, max: 1 };

dashboard.getRange("A21:H21").merge();
dashboard.getRange("A21").values = [["测试结论与可追溯信息"]];
dashboard.getRange("A21:H21").format.fill = colors.navy;
dashboard.getRange("A21:H21").format.font = { bold: true, color: colors.white };
dashboard.getRange("A22:H31").values = [
  ["最终严格回归", `${finalSummary.passed_cases_python}/${finalSummary.total_cases} 组通过；52/52 张牌面均覆盖。`, "", "", "", "", "", ""],
  ["算法改进前", `${priorSummary.passed_cases_python}/${priorSummary.total_cases} 组通过；增强矩形完整度、候选间距回退和搜索重叠容差后达到 ${finalSummary.passed_cases_python}/${finalSummary.total_cases}。`, "", "", "", "", "", ""],
  ["目标尺寸覆盖", `宽 ${finalSummary.target_dimension_range_mm.width_min.toFixed(2)}–${finalSummary.target_dimension_range_mm.width_max.toFixed(2)} mm；高 ${finalSummary.target_dimension_range_mm.height_min.toFixed(2)}–${finalSummary.target_dimension_range_mm.height_max.toFixed(2)} mm。`, "", "", "", "", "", ""],
  ["碎片数覆盖", `2 片 ${finalSummary.piece_count_distribution["2"]} 组；3 片 ${finalSummary.piece_count_distribution["3"]} 组；4 片 ${finalSummary.piece_count_distribution["4"]} 组。`, "", "", "", "", "", ""],
  ["异常归档", `${archiveSummary.prior_failed_cases} 组首轮失败、${archiveSummary.final_near_threshold_cases} 组最终临界通过、${archiveSummary.final_failed_cases} 组最终失败。`, "", "", "", "", "", ""],
  ["裁剪约束", "每片不超过 5 边、每边至少 20 mm、每片至少一条矩形外边；合成测试另设 4 mm 最小可观测顶点偏离，排除近共线退化交点。", "", "", "", "", "", ""],
  ["测试性质", "这是可重复的合成视觉回归；实机仍需镜头标定、光照/高度/阴影/畸变测试。", "", "", "", "", "", ""],
  ["牌面素材", "https://github.com/hayeah/playing-cards-assets", "", "", "", "", "", ""],
  ["结果目录", finalDir, "", "", "", "", "", ""],
  ["异常目录", archiveDir, "", "", "", "", "", ""],
];
for (let row = 22; row <= 31; row += 1) {
  dashboard.getRange(`B${row}:H${row}`).merge();
}
dashboard.getRange("A22:H31").format.borders = {
  preset: "all",
  style: "thin",
  color: colors.gray300,
};
dashboard.getRange("A22:A31").format.fill = colors.gray100;
dashboard.getRange("A22:A31").format.font = { bold: true, color: colors.navy };
dashboard.getRange("B22:H31").format.wrapText = true;
dashboard.getRange("A:Q").format.columnWidth = 13;
dashboard.getRange("A:A").format.columnWidth = 27;
dashboard.getRange("B:H").format.columnWidth = 14;
dashboard.freezePanes.freezeRows(2);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const imagePreviewBeforeExport = await workbook.render({
  sheetName: "截图索引",
  range: "A1:J7",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(previewDir, "截图索引_导出前.png"),
  new Uint8Array(await imagePreviewBeforeExport.arrayBuffer()),
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

// Re-open the exported workbook and verify data, formulas, drawings, and visual layout.
const exportedBlob = await FileBlob.load(outputPath);
const verifiedWorkbook = await SpreadsheetFile.importXlsx(exportedBlob);
const inspectSummary = await verifiedWorkbook.inspect({
  kind: "workbook,sheet,table,drawing",
  maxChars: 9000,
  tableMaxRows: 4,
  tableMaxCols: 8,
});
await fs.writeFile(
  path.join(outputDir, "inspect_summary.ndjson"),
  inspectSummary.ndjson,
  "utf8",
);
const formulaErrors = await verifiedWorkbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(outputDir, "formula_error_scan.ndjson"),
  formulaErrors.ndjson,
  "utf8",
);

const previewSpecs = [
  ["数据概览", "A1:Q31"],
  ["严格阈值", "A1:F14"],
  ["组级结果", "A1:N10"],
  ["单片误差", "A1:N10"],
  ["拼接边误差", "A1:N10"],
  ["52牌面汇总", `A1:${cardLastLetter}12`],
  ["截图索引", "A1:J7"],
  ["异常样本", "A1:K12"],
];
for (const [sheetName, range] of previewSpecs) {
  console.log(`rendering ${sheetName} ${range}`);
  const preview = await verifiedWorkbook.render({
    sheetName,
    range,
    scale: 1,
    format: "png",
  });
  const bytes = new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), bytes);
}

console.log(
  JSON.stringify(
    {
      outputPath,
      sheets: previewSpecs.map(([name]) => name),
      cases: cases.length,
      pieces: pieces.length,
      seams: seams.length,
      embeddedScreenshots: cases.length,
      anomalyRows: anomalyRows.length,
      previews: previewSpecs.length,
    },
    null,
    2,
  ),
);
