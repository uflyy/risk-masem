import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Share2,
  Table as TableIcon,
  Network,
  Calculator,
  CheckCircle2,
  Download,
  ChevronRight,
  ChevronLeft,
  AlertCircle,
  Settings2,
  BookOpen,
  Info,
  Save,
  Trash2,
  Undo2,
  EyeOff,
  RotateCcw,
  MousePointer2
} from "lucide-react";

// --- 1. 类型定义 ---

type VarName = string;

type Cell = {
  r: number;
  n: number;
};

type CellMatrix = Record<VarName, Record<VarName, Cell>>;

type Edge = { from: VarName; to: VarName };

type Coef = { 
  from: VarName; 
  to: VarName; 
  beta: number; 
  se: number;      
  t: number;       
  p: number;       
  vif: number;     
};

type Fit = {
  totalN: number;
  N_method: "harmonic" | "min";
  SRMR: number;
  df: number;
  observedMoments: number;
  freeParams: number;
  endogenousCount: number;
  chi2?: number;
  rmsea?: number;
  cfi?: number;
  tli?: number;
  aic?: number;
  bic?: number;
};

type EstResult = {
  coeffs: Coef[];
  r2: Record<VarName, number>;
  resid: Record<VarName, number>;
  fit: Fit;
  timestamp: number;
};

type SavedModel = {
  id: string;
  name: string;
  edges: Edge[];
  activeVars: VarName[];
  result: EstResult;
  fit: Fit; // [FIX] 显式提升 fit 到顶层，防止读取 undefined 导致白屏
};

type NodePos = Record<VarName, { x: number; y: number }>;

// --- 2. 常量与预设数据 ---

const BASE_VARS_7 = [
  "Cognitive Risk",
  "Affective Risk",
  "Efficacy Beliefs",
  "Travel Intention",
  "Travel Avoidance",
  "Risk Reduction",
  "Other Behavior"
];

// 左侧变量组 (用于自动布局)
const LEFT_SIDE_VARS = ["Cognitive Risk", "Affective Risk", "Efficacy Beliefs"];

// 论文数据预设 (Liu-Lastres et al., 2024)
const PRESET_MATRIX_DATA: Record<string, { r: number; n: number }> = {
  "Cognitive Risk|Affective Risk": { r: 0.479, n: 13851 },
  "Cognitive Risk|Efficacy Beliefs": { r: 0.212, n: 2655 },
  "Cognitive Risk|Travel Intention": { r: -0.175, n: 62268 },
  "Cognitive Risk|Travel Avoidance": { r: 0.347, n: 3109 },
  "Cognitive Risk|Risk Reduction": { r: 0.398, n: 26466 },
  "Cognitive Risk|Other Behavior": { r: -0.094, n: 4654 },
  "Affective Risk|Efficacy Beliefs": { r: -0.037, n: 11799 },
  "Affective Risk|Travel Intention": { r: 0.007, n: 1256 },
  "Affective Risk|Travel Avoidance": { r: 0.155, n: 1256 },
  "Affective Risk|Risk Reduction": { r: 0.356, n: 5346 },
  "Affective Risk|Other Behavior": { r: -0.273, n: 40521 },
  "Efficacy Beliefs|Travel Intention": { r: 0.389, n: 11500 },
  "Efficacy Beliefs|Travel Avoidance": { r: -0.070, n: 4959 },
  "Efficacy Beliefs|Risk Reduction": { r: 0.482, n: 31665 },
  "Efficacy Beliefs|Other Behavior": { r: 0.217, n: 2324 },
  "Travel Intention|Travel Avoidance": { r: -0.062, n: 1799 },
  "Travel Intention|Risk Reduction": { r: 0.167, n: 9475 },
  "Travel Intention|Other Behavior": { r: 0.176, n: 4734 },
  "Travel Avoidance|Risk Reduction": { r: 0.388, n: 1235 },
  "Travel Avoidance|Other Behavior": { r: 0.453, n: 1007 },
  "Risk Reduction|Other Behavior": { r: 0.308, n: 2812 },
};

// --- 3. 辅助函数 ---

// [Safe Format] 防止 null/undefined 导致 toFixed 报错
function safeFmt(num: number | undefined | null, digits: number = 3): string {
  if (num === undefined || num === null || isNaN(num) || !Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function deepClone<T>(obj: T): T {
  try {
    return JSON.parse(JSON.stringify(obj));
  } catch (e) {
    console.error("Deep clone failed", e);
    return obj; 
  }
}

function getNormalPValue(z: number): number {
  const absZ = Math.abs(z);
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const t = 1.0 / (1.0 + p * absZ);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absZ * absZ);
  return (1.0 - y) * 2;
}

function getSignificanceStar(p: number) {
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "";
}

// --- 4. 矩阵运算核心 (Matrix Algebra) ---

function matIdentity(n: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}
function matClone(A: number[][]): number[][] {
  return A.map((r) => r.slice());
}
function transpose(A: number[][]): number[][] {
  return A[0].map((_, j) => A.map((row) => row[j]));
}
function matMul(A: number[][], B: number[][]): number[][] {
  const n = A.length, m = B[0].length, k = B.length;
  const out = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
      out[i][j] = s;
    }
  }
  return out;
}
function matVecMul(A: number[][], v: number[]): number[] {
  return A.map((row) => row.reduce((s, x, i) => s + x * v[i], 0));
}
function vecDot(a: number[], b: number[]): number {
  return a.reduce((s, x, i) => s + x * b[i], 0);
}
function matInverse(A: number[][]): number[][] {
  const n = A.length;
  let M = matClone(A);
  let I = matIdentity(n);
  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    for (let r = col; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivotRow][col])) pivotRow = r;
    }
    if (Math.abs(M[pivotRow][col]) < 1e-12) throw new Error("Matrix is singular (cannot invert).");
    [M[col], M[pivotRow]] = [M[pivotRow], M[col]];
    [I[col], I[pivotRow]] = [I[pivotRow], I[col]];
    const piv = M[col][col];
    for (let j = 0; j < n; j++) { M[col][j] /= piv; I[col][j] /= piv; }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let j = 0; j < n; j++) { M[r][j] -= factor * M[col][j]; I[r][j] -= factor * I[col][j]; }
    }
  }
  return I;
}
function matDet(A: number[][]): number {
  const n = A.length;
  let M = matClone(A);
  let det = 1;
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let r = i; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[pivot][i])) pivot = r;
    if (Math.abs(M[pivot][i]) < 1e-12) return 0;
    if (pivot !== i) { [M[i], M[pivot]] = [M[pivot], M[i]]; det *= -1; }
    det *= M[i][i];
    const piv = M[i][i];
    for (let r = i + 1; r < n; r++) {
      const f = M[r][i] / piv;
      for (let c = i; c < n; c++) M[r][c] -= f * M[i][c];
    }
  }
  return det;
}
function matTrace(A: number[][]): number {
  return A.reduce((s, row, i) => s + row[i], 0);
}

// --- 5. 统计估算逻辑 ---

function makeEmptyCellMatrix(vars: VarName[]): CellMatrix {
  const M: CellMatrix = {};
  for (const r of vars) {
    M[r] = {};
    for (const c of vars) {
      if (r === c) M[r][c] = { r: 1, n: Number.NaN };
      else M[r][c] = { r: Number.NaN, n: Number.NaN };
    }
  }
  return M;
}

function computeTotalN(vars: VarName[], M: CellMatrix): number {
  const ns: number[] = [];
  for (let i = 0; i < vars.length; i++) {
    for (let j = i + 1; j < vars.length; j++) {
      const n1 = M[vars[i]]?.[vars[j]]?.n;
      const n2 = M[vars[j]]?.[vars[i]]?.n;
      const val = (Number.isFinite(n1) ? n1 : n2);
      if (Number.isFinite(val)) ns.push(val!);
    }
  }
  if (ns.length === 0) return Number.NaN;
  let denom = 0;
  for (const n of ns) denom += 1 / n;
  return ns.length / denom;
}

function buildRMatrix(vars: VarName[], M: CellMatrix): number[][] {
  return vars.map((r) => vars.map((c) => M[r][c].r));
}

function estimatePaths(vars: VarName[], cellM: CellMatrix, edges: Edge[], totalN: number): {
  coeffs: Coef[]; r2: Record<VarName, number>; resid: Record<VarName, number>;
} {
  const coeffs: Coef[] = [];
  const r2: Record<VarName, number> = {};
  const resid: Record<VarName, number> = {};
  const endogenous = new Set(edges.map((e) => e.to));

  for (const y of vars) {
    const parents = edges.filter((e) => e.to === y).map((e) => e.from);
    if (parents.length === 0) continue;

    const Rxx = parents.map((a) => parents.map((b) => cellM[a][b].r));
    const rXy = parents.map((a) => cellM[a][y].r);
    const inv = matInverse(Rxx);
    const beta = matVecMul(inv, rXy);
    const R2 = vecDot(rXy, beta);
    
    r2[y] = R2;
    resid[y] = Math.max(1e-8, 1 - R2);

    const k = parents.length;
    const df = totalN - k - 1;
    const errorVar = (1 - R2) / df;

    parents.forEach((x, i) => {
      const vif = inv[i][i];
      const se = Math.sqrt(errorVar * vif);
      const t = Math.abs(beta[i] / se);
      const p = getNormalPValue(t);
      coeffs.push({ from: x, to: y, beta: beta[i], se, t, p, vif });
    });
  }
  for (const v of vars) if (endogenous.has(v) && !(v in resid)) resid[v] = 1;
  return { coeffs, r2, resid };
}

function impliedSigma(vars: VarName[], S: number[][], edges: Edge[], coeffs: Coef[], resid: Record<VarName, number>): number[][] {
  const p = vars.length;
  const idx: Record<VarName, number> = Object.fromEntries(vars.map((v, i) => [v, i]));
  const endogenous = new Set(edges.map((e) => e.to));
  const B = Array.from({ length: p }, () => Array(p).fill(0));
  for (const e of coeffs) B[idx[e.to]][idx[e.from]] = e.beta;
  const exoIdx = vars.map((v, i) => (!endogenous.has(v) ? i : -1)).filter((i) => i >= 0);
  const Psi = Array.from({ length: p }, () => Array(p).fill(0));
  for (let a = 0; a < exoIdx.length; a++) {
    for (let b = 0; b < exoIdx.length; b++) {
      Psi[exoIdx[a]][exoIdx[b]] = S[exoIdx[a]][exoIdx[b]];
    }
  }
  for (const v of vars) if (endogenous.has(v)) Psi[idx[v]][idx[v]] = Math.max(1e-8, resid[v] ?? 1);
  const I = matIdentity(p);
  const IminusB = I.map((row, i) => row.map((x, j) => x - B[i][j]));
  const inv = matInverse(IminusB);
  const invT = transpose(inv);
  return matMul(matMul(inv, Psi), invT);
}

function calculateFit(vars: VarName[], S: number[][], Sigma: number[][], N: number, edges: Edge[]): Fit {
  const p = S.length;
  const detS = matDet(S);
  const detSig = matDet(Sigma);
  
  let sumSqDiff = 0, k_off = 0;
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      sumSqDiff += (S[i][j] - Sigma[i][j]) ** 2;
      k_off++;
    }
  }
  const srmr = Math.sqrt(sumSqDiff / Math.max(1, k_off));

  const observedMoments = (p * (p + 1)) / 2;
  const endoVars = new Set(edges.map(e => e.to));
  const freeParams = edges.length + endoVars.size; 
  const df = Math.max(0, observedMoments - freeParams);

  const SigInv = matInverse(Sigma);
  const tr = matTrace(matMul(S, SigInv));
  const Fml = Math.log(detSig) + tr - Math.log(detS) - p;
  const chi2 = (N - 1) * Fml;

  const Sigma0 = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? S[i][i] : 0)));
  const det0 = matDet(Sigma0);
  const inv0 = matInverse(Sigma0);
  const tr0 = matTrace(matMul(S, inv0));
  const F0 = Math.log(det0) + tr0 - Math.log(detS) - p;
  const chi2_0 = (N - 1) * F0;
  const df0 = (p * (p - 1)) / 2; 

  const cfi = 1 - Math.max(0, chi2 - df) / Math.max(1e-12, chi2_0 - df0);
  const tli = 1 - (chi2 / Math.max(1e-12, df) - 1) / (chi2_0 / Math.max(1e-12, df0) - 1);
  const rmsea = Math.sqrt(Math.max(0, (chi2 - df) / (Math.max(1e-12, df) * (N - 1))));

  const term = p * 1.837877 + Math.log(detSig) + tr; 
  const minus2LL = N * term;
  
  const exoVars = vars.filter(v => !endoVars.has(v));
  const exoCovs = (exoVars.length * (exoVars.length + 1)) / 2; 
  const q = edges.length + endoVars.size + exoCovs; 

  const aic = minus2LL + 2 * q;
  const bic = minus2LL + q * Math.log(N);

  return { 
    totalN: N, N_method: "harmonic", SRMR: srmr, df, observedMoments, 
    freeParams, endogenousCount: endoVars.size, 
    chi2, rmsea, cfi, tli, aic, bic 
  };
}

// --- 6. 主组件 ---

export default function COVIDPathModelBuilder() {
  const [step, setStep] = useState(0);
  
  // 状态管理
  const [activeVars, setActiveVars] = useState<VarName[]>([...BASE_VARS_7]);
  const [hiddenVars, setHiddenVars] = useState<VarName[]>([]);
  
  // 数据
  const [cellM, setCellM] = useState<CellMatrix>(() => {
    const m = makeEmptyCellMatrix(BASE_VARS_7);
    for(const [key, val] of Object.entries(PRESET_MATRIX_DATA)) {
      const [a, b] = key.split("|");
      if (m[a] && m[b]) {
        m[a][b] = { ...val };
        m[b][a] = { ...val };
      }
    }
    for(const v of BASE_VARS_7) m[v][v] = { r: 1, n: Number.NaN };
    return m;
  });

  // 模型
  const [edges, setEdges] = useState<Edge[]>([]);
  const [nodePos, setNodePos] = useState<NodePos>({});
  const [connectFrom, setConnectFrom] = useState<VarName | null>(null);

  // 结果 & 保存
  const [lastEst, setLastEst] = useState<EstResult | null>(null);
  const [savedModels, setSavedModels] = useState<SavedModel[]>([]);
  const [modelNameInput, setModelNameInput] = useState("");
  const [estError, setEstError] = useState<string | null>(null);

  // --- 自动布局逻辑 (Auto Layout) ---
  useEffect(() => {
    const newPos: NodePos = {};
    const leftVars = activeVars.filter(v => LEFT_SIDE_VARS.includes(v));
    const rightVars = activeVars.filter(v => !LEFT_SIDE_VARS.includes(v));

    // Left Column (x=200)
    const leftSpacing = 400 / (leftVars.length + 1);
    leftVars.forEach((v, i) => {
      newPos[v] = { x: 200, y: 50 + leftSpacing * (i + 1) };
    });

    // Right Column (x=600)
    const rightSpacing = 400 / (rightVars.length + 1);
    rightVars.forEach((v, i) => {
      newPos[v] = { x: 600, y: 50 + rightSpacing * (i + 1) };
    });

    setNodePos(newPos);
  }, [activeVars.length]); // 仅当变量数量变化时重置

  // Actions
  const toggleVar = (v: VarName) => {
    if (activeVars.includes(v)) {
      setActiveVars(prev => prev.filter(x => x !== v));
      setHiddenVars(prev => [...prev, v]);
      setEdges(prev => prev.filter(e => e.from !== v && e.to !== v));
    } else {
      setHiddenVars(prev => prev.filter(x => x !== v));
      setActiveVars(prev => [...prev, v]);
    }
    setLastEst(null);
  };

  const setCell = (r: VarName, c: VarName, val: Partial<Cell>) => {
    setCellM(prev => {
      const next = deepClone(prev);
      next[r][c] = { ...next[r][c], ...val };
      next[c][r] = { ...next[c][r], ...val };
      return next;
    });
    setLastEst(null);
  };

  const runEstimation = () => {
    setEstError(null);
    if (edges.length === 0) {
      setEstError("Please create at least one path.");
      return;
    }
    
    for(const r of activeVars) {
      for(const c of activeVars) {
        if (r !== c) {
          const cell = cellM[r][c];
          if (!cell || !Number.isFinite(cell.r) || !Number.isFinite(cell.n)) {
            setEstError(`Missing correlation or sample size for pair: ${r} - ${c}`);
            return;
          }
        }
      }
    }

    try {
      const N = computeTotalN(activeVars, cellM);
      const S = buildRMatrix(activeVars, cellM);
      const { coeffs, r2, resid } = estimatePaths(activeVars, cellM, edges, N);
      const Sigma = impliedSigma(activeVars, S, edges, coeffs, resid);
      const fit = calculateFit(activeVars, S, Sigma, N, edges);
      
      setLastEst({ coeffs, r2, resid, fit, timestamp: Date.now() });
      setStep(3);
    } catch (e: any) {
      setEstError(e.message);
    }
  };

  const saveCurrentModel = () => {
    if (!lastEst) return;
    const name = modelNameInput.trim() || `Model ${savedModels.length + 1}`;
    
    // [FIX]: 显式提取 fit 到第一层，避免渲染表格时读取 undefined
    const newModel: SavedModel = {
      id: Date.now().toString(),
      name,
      edges: deepClone(edges),
      activeVars: deepClone(activeVars),
      result: deepClone(lastEst),
      fit: deepClone(lastEst.fit) 
    };

    if (savedModels.length >= 10) {
      alert("Maximum 10 models allowed. Please delete one to save new.");
      return;
    }
    setSavedModels([...savedModels, newModel]);
    setModelNameInput("");
  };

  const restoreModel = (m: SavedModel) => {
    if (!confirm(`Restore "${m.name}"? Unsaved progress will be lost.`)) return;
    
    setActiveVars(m.activeVars);
    const newHidden = BASE_VARS_7.filter(v => !m.activeVars.includes(v));
    setHiddenVars(newHidden);
    
    setEdges(m.edges);
    setLastEst(m.result);
    setStep(3);
  };

  // --- SVG 交互 (Safe Drag & Click) ---
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<{v:string, dx:number, dy:number} | null>(null);

  const handlePointerDown = (e: React.PointerEvent, v: string) => {
    if(step !== 2) return;
    e.preventDefault();
    e.stopPropagation();
    
    const svg = svgRef.current;
    if (!svg) return;

    try {
      const CTM = svg.getScreenCTM();
      if (!CTM) return;
      
      const pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const cursor = pt.matrixTransform(CTM.inverse());
      
      const pos = nodePos[v];
      if (pos) {
        dragRef.current = { v, dx: pos.x - cursor.x, dy: pos.y - cursor.y };
      }
    } catch (err) {
      // ignore
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const svg = svgRef.current;
    if (!svg) return;

    try {
      const CTM = svg.getScreenCTM();
      if (!CTM) return;

      const pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const cursor = pt.matrixTransform(CTM.inverse());
      
      const newX = Math.max(50, Math.min(750, cursor.x + dragRef.current.dx));
      const newY = Math.max(50, Math.min(400, cursor.y + dragRef.current.dy));

      setNodePos(prev => ({
        ...prev,
        [dragRef.current!.v]: { x: newX, y: newY }
      }));
    } catch (err) {
      // ignore
    }
  };

  const handleNodeClick = (v: string) => {
    if (step !== 2) return;
    if (dragRef.current) {
        dragRef.current = null;
        return;
    }

    if (!connectFrom) {
      setConnectFrom(v);
    } else {
      if (connectFrom !== v) {
        // Toggle
        const exists = edges.find(e => e.from === connectFrom && e.to === v);
        if (exists) {
          setEdges(edges.filter(e => e !== exists));
        } else {
          setEdges([...edges, { from: connectFrom, to: v }]);
        }
      }
      setConnectFrom(null);
    }
  };

  // --- 贝塞尔曲线渲染 (Curved Paths) ---
  const renderDiagram = (readOnly: boolean, results?: EstResult) => {
    return (
      <svg 
        ref={!readOnly ? svgRef : undefined}
        viewBox="0 0 800 450" 
        className={`w-full h-full ${readOnly ? '' : 'cursor-crosshair'}`}
        onPointerMove={!readOnly ? handlePointerMove : undefined} 
        onPointerUp={() => dragRef.current = null}
        onPointerLeave={() => dragRef.current = null}
      >
        <defs>
          <marker id={readOnly ? "arrow-read" : "arrow"} markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill={readOnly ? "#475569" : "#64748b"} />
          </marker>
        </defs>
        
        {edges.map((e, i) => {
          const p1 = nodePos[e.from];
          const p2 = nodePos[e.to];
          
          if (!p1 || !p2) return null; 

          // 曲线逻辑
          const mx = (p1.x + p2.x) / 2;
          const my = (p1.y + p2.y) / 2;
          
          // 如果节点在同一列 (x坐标相近)，增加弯曲度以避让
          const isVertical = Math.abs(p1.x - p2.x) < 50;
          // 如果跨度大，也增加弯曲
          const dist = Math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2);
          const offset = isVertical ? 60 : (dist > 300 ? 40 : 20); 
          
          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const len = Math.sqrt(dx*dx + dy*dy) || 1;
          const udx = dx/len;
          const udy = dy/len;
          
          const cx = mx - udy * offset;
          const cy = my + udx * offset;

          const pathD = `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`;

          // Label Position (Bezier t=0.5)
          const labelX = 0.25*p1.x + 0.5*cx + 0.25*p2.x;
          const labelY = 0.25*p1.y + 0.5*cy + 0.25*p2.y;

          let label = null;
          if (readOnly && results) {
            const coef = results.coeffs.find(c => c.from === e.from && c.to === e.to);
            if (coef) label = `${coef.beta.toFixed(2)}${getSignificanceStar(coef.p)}`;
          }

          return (
            <g key={i} onClick={!readOnly ? (ev) => { ev.stopPropagation(); setEdges(edges.filter((_, idx) => idx !== i)); } : undefined} className={!readOnly ? "cursor-pointer group" : ""}>
              <path d={pathD} stroke={readOnly ? "#475569" : "#94a3b8"} strokeWidth="2" fill="none" markerEnd={`url(#${readOnly ? 'arrow-read' : 'arrow'})`} className={!readOnly ? "group-hover:stroke-rose-500 transition-colors" : ""} />
              {!readOnly && <path d={pathD} stroke="transparent" strokeWidth="15" fill="none" />}
              
              {label && (
                <g>
                  <rect x={labelX - 18} y={labelY - 9} width={36} height={18} fill="white" stroke="#e2e8f0" rx="4" />
                  <text x={labelX} y={labelY} dy="3" textAnchor="middle" fontSize="10" fontWeight="bold" fill="#1e293b">{label}</text>
                </g>
              )}
            </g>
          );
        })}

        {activeVars.map(v => {
          const p = nodePos[v];
          if (!p) return null;
          
          const isSrc = connectFrom === v;
          let r2Label = null;
          if (readOnly && results && results.r2[v] !== undefined && results.r2[v] > 0.001) {
            r2Label = `R²=${results.r2[v].toFixed(2)}`;
          }

          return (
            <g key={v} transform={`translate(${p.x},${p.y})`} 
               onPointerDown={!readOnly ? (e) => handlePointerDown(e, v) : undefined}
               onClick={!readOnly ? (e) => { e.stopPropagation(); handleNodeClick(v); } : undefined}
               className={!readOnly ? "cursor-grab active:cursor-grabbing" : ""}
            >
              <circle r="40" className={`fill-white stroke-2 transition-all ${isSrc ? 'stroke-indigo-500 fill-indigo-50' : 'stroke-slate-300 hover:stroke-indigo-400'}`} />
              <text dy="0" className="text-[10px] font-bold fill-slate-700 pointer-events-none text-center" textAnchor="middle">
                {v.split(' ').map((line, i) => <tspan key={i} x="0" dy={i===0 ? "-5" : "12"}>{line}</tspan>)}
              </text>
              {r2Label && (
                <text dy="-50" textAnchor="middle" fontSize="10" fontWeight="bold" fill="#4f46e5">{r2Label}</text>
              )}
            </g>
          )
        })}
      </svg>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-800 pb-24">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Share2 className="text-indigo-600" size={24} />
            <div>
              <h1 className="font-bold text-lg leading-tight text-slate-900">MASEM of Risk Perception and Travel Behavior</h1>
              <p className="text-[10px] text-slate-500 font-medium">
                Programmed by Dr. Yang Yang, Temple University; Based on Liu-Lastres, Yang and Zhang (2024)
              </p>
            </div>
          </div>
          <div className="flex gap-1">
            {[0,1,2,3].map(i => (
              <div key={i} className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold ${step===i ? 'bg-indigo-100 text-indigo-700' : 'text-slate-400'}`}>
                <span className={`w-5 h-5 flex items-center justify-center rounded-full ${step===i?'bg-indigo-600 text-white':'bg-slate-200'}`}>{i+1}</span>
                {['Variables', 'Data', 'Model', 'Results'][i]}
              </div>
            ))}
          </div>
          <a
            href="./Risk_MASEM_Manual.pdf" 
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-bold text-indigo-700 bg-indigo-50 border border-indigo-100 rounded-lg hover:bg-indigo-100 transition-all shadow-sm"
          >
            <Download size={14} />
            Download Manual
          </a>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        
        {/* Step 0: Variables */}
        {step === 0 && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 max-w-3xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold mb-2">Model Configuration</h2>
              <p className="text-slate-500">Manage the 7 variables from the paper. You can hide irrelevant ones.</p>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-emerald-500"/> Active Variables
                </h3>
                <div className="flex flex-wrap gap-2">
                  {activeVars.map(v => (
                    <div key={v} className="group flex items-center gap-2 px-3 py-2 bg-white border border-slate-200 rounded-lg shadow-sm hover:border-rose-200 transition-colors">
                      <span className="text-sm font-medium text-slate-700">{v}</span>
                      <button onClick={() => toggleVar(v)} className="text-slate-400 hover:text-rose-500" title="Hide variable">
                        <EyeOff size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {hiddenVars.length > 0 && (
                <div className="pt-4 border-t border-slate-100">
                  <h3 className="text-sm font-bold text-slate-400 mb-3 flex items-center gap-2">
                    <Trash2 size={16}/> Hidden Variables
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {hiddenVars.map(v => (
                      <div key={v} className="flex items-center gap-2 px-3 py-2 bg-slate-50 border border-slate-200 border-dashed rounded-lg opacity-70">
                        <span className="text-sm text-slate-500 line-through">{v}</span>
                        <button onClick={() => toggleVar(v)} className="text-indigo-500 hover:text-indigo-700 font-bold text-xs flex items-center gap-1">
                          <Undo2 size={12}/> Restore
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Step 1: Data */}
        {step === 1 && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="mb-4 flex items-start gap-3 bg-indigo-50 p-4 rounded-xl text-indigo-900 text-sm">
              <Info className="shrink-0 mt-0.5 text-indigo-600" size={18} />
              <div>
                <strong>Pre-loaded Data:</strong> Correlation matrix and sample sizes are pre-loaded from <em>Liu-Lastres et al. (2024), Table 4</em>.
                <br/>Values are editable if you need to adjust or correct specific entries.
              </div>
            </div>

            <div className="overflow-x-auto pb-4">
              <table className="w-full text-xs border-collapse min-w-[800px]">
                <thead>
                  <tr>
                    <th className="p-3 text-left bg-slate-50 border-b">Matrix (r | N)</th>
                    {activeVars.map(v => <th key={v} className="p-2 border-b bg-slate-50 text-slate-600 font-bold min-w-[120px]">{v}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {activeVars.map(r => (
                    <tr key={r} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="p-3 font-bold text-slate-700 bg-slate-50 border-r">{r}</td>
                      {activeVars.map(c => {
                        const isDiag = r === c;
                        const cell = cellM[r]?.[c] || { r: NaN, n: NaN };
                        return (
                          <td key={c} className="p-2 border-r border-slate-50">
                            {isDiag ? (
                              <div className="text-center text-slate-300 font-mono">1.00</div>
                            ) : (
                              <div className="flex gap-1">
                                <div className="relative flex-1">
                                  <span className="absolute left-1.5 top-1.5 text-[9px] text-slate-400 font-bold">r</span>
                                  <input 
                                    type="number" step="0.01" 
                                    className="w-full pl-4 pr-1 py-1 bg-white border border-slate-200 rounded text-slate-700 font-mono focus:border-indigo-500 outline-none"
                                    value={cell.r}
                                    onChange={e => setCell(r, c, { r: parseFloat(e.target.value) })}
                                  />
                                </div>
                                <div className="relative flex-1">
                                  <span className="absolute left-1.5 top-1.5 text-[9px] text-slate-400 font-bold">N</span>
                                  <input 
                                    type="number" 
                                    className="w-full pl-4 pr-1 py-1 bg-slate-50 border border-slate-200 rounded text-slate-500 font-mono focus:border-indigo-500 outline-none"
                                    value={cell.n}
                                    onChange={e => setCell(r, c, { n: parseFloat(e.target.value) })}
                                  />
                                </div>
                              </div>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Step 2: Model */}
        {step === 2 && (
          <div className="h-[600px] bg-white rounded-2xl shadow-sm border border-slate-200 flex flex-col">
            <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
              <div className="flex items-center gap-4 text-sm text-slate-600">
                <div className="flex items-center gap-2 px-3 py-1 bg-white border border-slate-200 rounded-lg shadow-sm">
                  <MousePointer2 size={16} className="text-indigo-600"/>
                  <span>Click two nodes to connect</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1 bg-white border border-slate-200 rounded-lg shadow-sm">
                  <Network size={16} className="text-indigo-600"/>
                  <span>Drag nodes to arrange</span>
                </div>
                {connectFrom && (
                  <span className="flex items-center gap-2 text-indigo-600 font-bold animate-pulse">
                    <div className="w-2 h-2 rounded-full bg-indigo-600"></div>
                    Connecting from {connectFrom}...
                  </span>
                )}
              </div>
              <button onClick={() => setEdges([])} className="text-xs text-rose-600 hover:bg-rose-50 px-3 py-1 rounded border border-rose-100 hover:border-rose-200 transition-all">
                Clear All Paths
              </button>
            </div>
            
            <div className="flex-1 relative bg-slate-50 overflow-hidden">
              {renderDiagram(false)}
            </div>
          </div>
        )}

        {/* Step 3: Results */}
        {step === 3 && lastEst && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4">
            
            {/* Result Diagram */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="px-6 py-3 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
                <h3 className="font-bold text-slate-700 flex items-center gap-2"><Network size={18}/> Path Result Diagram</h3>
                <span className="text-xs text-slate-400">Standardized Beta on paths, R² on nodes</span>
              </div>
              <div className="h-[500px] relative">
                {renderDiagram(true, lastEst)}
              </div>
            </div>

            {/* Header / Save */}
            <div className="flex flex-col md:flex-row justify-between items-end gap-4 bg-indigo-50 p-6 rounded-2xl border border-indigo-100">
              <div>
                <h2 className="text-xl font-bold text-indigo-900 flex items-center gap-2">
                  <Calculator size={20}/> Estimation Tables
                </h2>
                <p className="text-indigo-700 text-sm mt-1">
                  Harmonic Mean N = <span className="font-mono font-bold">{lastEst.fit.totalN.toFixed(0)}</span>
                </p>
              </div>
              <div className="flex gap-2 w-full md:w-auto">
                <input 
                  type="text" 
                  placeholder="Model Name (e.g. Model 1)" 
                  className="px-4 py-2 rounded-xl border border-indigo-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 w-full"
                  value={modelNameInput}
                  onChange={e => setModelNameInput(e.target.value)}
                />
                <button 
                  onClick={saveCurrentModel}
                  className="flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-xl font-bold hover:bg-indigo-700 shadow-md hover:shadow-lg transition-all"
                >
                  <Save size={18}/> Save Model
                </button>
              </div>
            </div>

            {/* Fit Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
              {[
                { l: "Chi-Square", v: safeFmt(lastEst.fit.chi2, 1) },
                { l: "df", v: lastEst.fit.df },
                { l: "RMSEA", v: safeFmt(lastEst.fit.rmsea), good: (lastEst.fit.rmsea||1) < 0.08 },
                { l: "SRMR", v: safeFmt(lastEst.fit.SRMR), good: lastEst.fit.SRMR < 0.08 },
                { l: "CFI", v: safeFmt(lastEst.fit.cfi), good: (lastEst.fit.cfi||0) > 0.9 },
                { l: "TLI", v: safeFmt(lastEst.fit.tli), good: (lastEst.fit.tli||0) > 0.9 },
                { l: "AIC", v: safeFmt(lastEst.fit.aic, 0) },
                { l: "BIC", v: safeFmt(lastEst.fit.bic, 0) },
              ].map((m, i) => (
                <div key={i} className={`p-3 rounded-xl border ${m.good===true ? 'bg-emerald-50 border-emerald-200' : m.good===false ? 'bg-rose-50 border-rose-200' : 'bg-white border-slate-200'}`}>
                  <div className="text-[10px] uppercase font-bold text-slate-400">{m.l}</div>
                  <div className="text-lg font-mono font-bold text-slate-700">{m.v}</div>
                </div>
              ))}
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              {/* Path Coeffs */}
              <div className="md:col-span-2 bg-white border border-slate-200 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 font-bold text-slate-700">Path Coefficients</div>
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 text-slate-500 text-xs uppercase">
                    <tr>
                      <th className="px-6 py-3 text-left">Path</th>
                      <th className="px-6 py-3 text-right">Beta</th>
                      <th className="px-6 py-3 text-right">S.E.</th>
                      <th className="px-6 py-3 text-right">p-value</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {lastEst.coeffs.map((c, i) => (
                      <tr key={i} className="hover:bg-slate-50">
                        <td className="px-6 py-3 font-medium">
                          {c.from} <span className="text-slate-300 mx-1">→</span> {c.to}
                        </td>
                        <td className="px-6 py-3 text-right font-mono font-bold text-indigo-600">
                          {c.beta.toFixed(3)}
                        </td>
                        <td className="px-6 py-3 text-right font-mono text-slate-500">{c.se.toFixed(3)}</td>
                        <td className="px-6 py-3 text-right font-mono">
                          {c.p < 0.001 ? <b className="text-emerald-600">&lt;.001***</b> : c.p.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* R-Squared */}
              <div className="bg-white border border-slate-200 rounded-xl p-6">
                <div className="font-bold text-slate-700 mb-4">R-Squared (R²)</div>
                <div className="space-y-3">
                  {Object.entries(lastEst.r2).map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-sm">
                      <span className="text-slate-600">{k}</span>
                      <span className="font-mono font-bold text-indigo-600 bg-indigo-50 px-2 py-1 rounded">{v.toFixed(3)}</span>
                    </div>
                  ))}
                  {Object.keys(lastEst.r2).length === 0 && <div className="text-slate-400 italic text-xs">No endogenous variables</div>}
                </div>
              </div>
            </div>

            {/* COMPARISON TABLE */}
            {savedModels.length > 0 && (
              <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-lg mt-8">
                <div className="bg-slate-800 text-white px-6 py-4 flex justify-between items-center">
                  <h3 className="font-bold flex items-center gap-2"><Settings2 size={18}/> Model Comparison</h3>
                  <span className="text-xs bg-slate-700 px-2 py-1 rounded text-slate-300">{savedModels.length} / 10 saved</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-slate-100 text-slate-600 border-b border-slate-200">
                      <tr>
                        <th className="px-4 py-3 min-w-[150px]">Model Name</th>
                        <th className="px-2 py-3 text-center">Paths</th>
                        <th className="px-2 py-3 text-right">Chi²</th>
                        <th className="px-2 py-3 text-right">CFI</th>
                        <th className="px-2 py-3 text-right">SRMR</th>
                        <th className="px-2 py-3 text-right bg-slate-200/50">AIC</th>
                        <th className="px-2 py-3 text-right bg-slate-200/50">BIC</th>
                        <th className="px-4 py-3 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {savedModels.map(m => (
                        <tr key={m.id} className="hover:bg-slate-50">
                          <td className="px-4 py-3 font-bold text-indigo-900">{m.name}</td>
                          <td className="px-2 py-3 text-center text-slate-500">{m.edges.length}</td>
                          {/* [FIX] 使用 safeFmt 和 m.fit 确保安全渲染 */}
                          <td className="px-2 py-3 text-right font-mono">{safeFmt(m.fit.chi2, 1)}</td>
                          <td className={`px-2 py-3 text-right font-mono ${(m.fit.cfi||0)>0.9?'text-emerald-600 font-bold':''}`}>{safeFmt(m.fit.cfi)}</td>
                          <td className="px-2 py-3 text-right font-mono">{safeFmt(m.fit.SRMR)}</td>
                          <td className="px-2 py-3 text-right font-mono bg-slate-50/50">{safeFmt(m.fit.aic, 0)}</td>
                          <td className="px-2 py-3 text-right font-mono bg-slate-50/50">{safeFmt(m.fit.bic, 0)}</td>
                          <td className="px-4 py-3 text-right flex justify-end gap-2">
                            <button onClick={() => restoreModel(m)} className="text-indigo-500 hover:text-indigo-700 flex items-center gap-1 text-xs font-bold bg-indigo-50 px-2 py-1 rounded">
                              <RotateCcw size={12}/> Restore
                            </button>
                            <button onClick={() => setSavedModels(prev => prev.filter(x => x.id !== m.id))} className="text-rose-400 hover:text-rose-600 p-1">
                              <Trash2 size={16}/>
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

          </div>
        )}

        {/* Footer Actions */}
        <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-4 z-50">
          <div className="max-w-7xl mx-auto flex justify-between items-center">
            <button 
              onClick={() => setStep(s => Math.max(0, s-1))}
              disabled={step === 0}
              className="px-6 py-2.5 rounded-xl font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
            >
              <ChevronLeft size={18}/> Back
            </button>

            {estError && <div className="text-rose-600 text-sm font-medium flex items-center gap-2"><AlertCircle size={16}/> {estError}</div>}

            <div className="flex gap-3">
              {step === 2 && (
                <button 
                  onClick={runEstimation}
                  className="px-6 py-2.5 rounded-xl font-bold bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-200 flex items-center gap-2 transition-all hover:scale-105 active:scale-95"
                >
                  Run Estimation <ChevronRight size={18}/>
                </button>
              )}
              {step < 3 && step !== 2 && (
                <button 
                  onClick={() => setStep(s => s+1)}
                  className="px-6 py-2.5 rounded-xl font-bold bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-200 flex items-center gap-2 transition-all hover:scale-105 active:scale-95"
                >
                  Next Step <ChevronRight size={18}/>
                </button>
              )}
              {step === 3 && (
                <button 
                  onClick={() => { setStep(2); }}
                  className="px-6 py-2.5 rounded-xl font-medium text-indigo-600 border border-indigo-200 hover:bg-indigo-50 transition-colors"
                >
                  Adjust Model
                </button>
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}