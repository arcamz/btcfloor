(() => {
  "use strict";

  const {createChart, CandlestickSeries, LineSeries, LineStyle, PriceScaleMode} = LightweightCharts;
  const data = JSON.parse(document.getElementById("market-data").textContent);
  const $ = id => document.getElementById(id);
  const day = 86_400_000;
  const isoDay = stamp => new Date(stamp).toISOString().slice(0, 10);
  const stamp = date => Date.parse(date + "T00:00:00Z");
  const monday = date => {
    const time = stamp(date);
    return isoDay(time - ((new Date(time).getUTCDay() + 6) % 7) * day);
  };
  const precision = new Intl.NumberFormat("en-US", {maximumFractionDigits: 2});
  const dollars = new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2,
  });
  const price = value => dollars.format(value);
  const paneNames = {
    price: "BTC/USD · candles", ratio: "Holder ratios", z: "Holder Z-score", btc: "Holder flows · BTC",
  };
  const selected = new Set(data.metrics.filter(metric => metric.default && metric.last).map(metric => metric.key));
  const daily = data.dates.map((time, index) => {
    const [open, high, low, close] = data.candles[index];
    return {candle: {time, open, high, low, close}, index, date: time, count: 1};
  });
  const weekly = [];
  for (const row of daily) {
    const {time, open, high, low, close} = row.candle;
    const week = monday(time);
    const current = weekly.at(-1);
    if (current?.candle.time === week) {
      current.candle.high = Math.max(current.candle.high, high);
      current.candle.low = Math.min(current.candle.low, low);
      current.candle.close = close;
      current.index = row.index;
      current.date = time;
      current.count++;
    } else {
      weekly.push({candle: {time: week, open, high, low, close}, index: row.index, date: time, count: 1});
    }
  }

  let chart, rows, byTime, paneKeys, active, overlayFrame;
  let logarithmic = false;
  let chosenRange = "365";
  const color = name => getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim();
  const swatch = name => {
    const dot = document.createElement("i");
    dot.className = "swatch";
    dot.style.setProperty("--series-color", "var(--" + name + ")");
    dot.setAttribute("aria-hidden", "true");
    return dot;
  };
  const formatter = pane => {
    if (pane === "price") return price;
    if (pane === "ratio") return value => precision.format(value) + "×";
    if (pane === "btc") return value => precision.format(value) + " BTC";
    return value => precision.format(value);
  };

  function checklist() {
    const groups = new Map();
    for (const metric of data.metrics) {
      if (!groups.has(metric.group)) {
        const field = document.createElement("fieldset");
        const legend = document.createElement("legend");
        legend.textContent = metric.group;
        field.append(legend);
        groups.set(metric.group, field);
        $("metric-options").append(field);
      }
      const label = document.createElement("label");
      label.className = "metric-choice";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = selected.has(metric.key);
      box.disabled = !metric.last;
      box.dataset.metric = metric.key;
      box.addEventListener("change", () => {
        if (box.checked) selected.add(metric.key);
        else selected.delete(metric.key);
        render(chart.timeScale().getVisibleLogicalRange());
      });
      const title = document.createElement("span");
      title.textContent = metric.label;
      const detail = document.createElement("small");
      detail.textContent = !metric.last ? "Unavailable" : metric.group === "Technical" ? "" : "Through " + metric.last;
      label.title = metric.last ? metric.source + " · " + metric.first + " through " + metric.last : metric.source;
      label.append(box, swatch(metric.color), title, detail);
      groups.get(metric.group).append(label);
    }
  }

  function readout(row) {
    const time = row?.candle.time;
    const isWeek = $("interval").value === "weekly";
    $("date-label").textContent = !row ? "BTC/USD · no candle"
      : isWeek ? "Week of " + time + " · through " + row.date + (row.count === 7 ? "" : " · partial")
      : "BTC/USD · " + time;
    $("close").textContent = row ? price(row.candle.close) : "—";
    $("ohlc").textContent = row ? ["open", "high", "low"].map(field =>
      field[0].toUpperCase() + " " + price(row.candle[field])).join(" · ") : "";
    const entries = [];
    for (const metric of active) {
      const item = document.createElement("span");
      item.className = "readout-item";
      const values = Object.values(metric.values).map(series => series[row?.index] ?? null);
      const shown = document.createElement("b");
      const fmt = formatter(metric.pane);
      const value = x => x === null ? "—" : fmt(x);
      if (metric.key === "bands") {
        item.append(swatch(metric.color), "EMA27 / Z");
        shown.textContent = value(values[0]) + " / " + (values[3] === null ? "—" : precision.format(values[3]));
        if (values[1] !== null) item.title = "Lower " + price(values[2]) + " · Upper " + price(values[1]);
      } else {
        item.append(swatch(metric.color), metric.label);
        shown.textContent = value(values[0]);
      }
      item.append(shown);
      entries.push(item);
    }
    $("readout").replaceChildren(...entries);
  }

  function drawOverlay() {
    overlayFrame = null;
    let top = 0;
    const elements = [];
    const paneHeights = chart.panes().map(pane => pane.getHeight());
    for (const [index, key] of paneKeys.entries()) {
      const label = document.createElement("span");
      label.className = "pane-label";
      label.textContent = paneNames[key];
      label.style.top = top + 5 + "px";
      elements.push(label);
      top += paneHeights[index];
    }
    if ($("cycles").checked) {
      const scale = chart.timeScale();
      for (const anchor of data.cycle_lows) {
        const date = $("interval").value === "weekly" ? monday(anchor.date) : anchor.date;
        const x = scale.timeToCoordinate(date);
        if (x === null || x < 30 || x > scale.width()) continue;
        const line = document.createElement("span");
        line.className = "cycle-line";
        line.style.cssText = "left:" + x + "px;height:" + top + "px";
        const label = document.createElement("span");
        label.className = "cycle-label";
        label.textContent = anchor.label;
        label.style.cssText = "left:" + x + "px;top:24px";
        elements.push(line, label);
      }
    }
    $("chart-overlay").replaceChildren(...elements);
  }

  function scheduleOverlay() {
    if (overlayFrame == null) overlayFrame = requestAnimationFrame(drawOverlay);
  }

  function fit(range = chosenRange) {
    chosenRange = range;
    const fromDate = stamp(rows.at(-1).candle.time) - (Number(range) - 1) * day;
    const from = range === "all" ? 0 : rows.findIndex(row => stamp(row.candle.time) >= fromDate);
    chart.timeScale().setVisibleLogicalRange({from: Math.max(0, from), to: rows.length - 1});
    for (const button of $("ranges").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.range === range));
    }
  }

  function metricData(metric, field) {
    const chunks = [];
    let segment = [];
    const values = metric.values[field];
    for (const {index, candle: {time}} of rows) {
      const value = values[index];
      if (value !== null && !(logarithmic && metric.pane === "price" && value <= 0)) {
        segment.push({time, value});
      } else if (segment.length) {
        chunks.push(segment);
        segment = [];
      }
    }
    if (segment.length) chunks.push(segment);
    return chunks;
  }

  function render(keepRange) {
    if (chart) chart.remove();
    const isWeek = $("interval").value === "weekly";
    rows = isWeek ? weekly : daily;
    byTime = new Map(rows.map(row => [row.candle.time, row]));
    active = data.metrics.filter(metric => selected.has(metric.key));
    paneKeys = Object.keys(paneNames).filter(key => key === "price" || active.some(m => m.pane === key));
    $("metric-count").textContent = active.length ? active.length + " selected" : "None selected";
    chart = createChart($("chart"), {
      autoSize: true,
      layout: {
        background: {color: color("plot")}, textColor: color("muted"), attributionLogo: true,
        panes: {separatorColor: color("border"), separatorHoverColor: color("hover")},
      },
      grid: {vertLines: {color: color("grid")}, horzLines: {color: color("grid")}},
      rightPriceScale: {minimumWidth: 86, borderColor: color("border")},
      timeScale: {borderColor: color("border"), minBarSpacing: 0.01, fixLeftEdge: true,
        lockVisibleTimeRangeOnResize: true, ignoreWhitespaceIndices: false},
      crosshair: {vertLine: {color: color("muted")}, horzLine: {color: color("muted")}},
    });
    $("chart").dataset.chartVersion = LightweightCharts.version();
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: color("up"), downColor: color("down"), wickUpColor: color("up"),
      wickDownColor: color("down"), borderVisible: false, priceLineVisible: false,
      priceFormat: {type: "custom", formatter: value => "$" + precision.format(value), minMove: 0.01},
    });
    candles.setData(rows.map(row => row.candle));
    for (const [index, key] of paneKeys.entries()) {
      if (index > 0) chart.addPane(true).setStretchFactor(1);
      for (const metric of active.filter(entry => entry.pane === key)) {
        const fields = metric.key === "bands" ? ["basis", "upper", "lower"] : [metric.key];
        for (const field of fields) {
          for (const points of metricData(metric, field)) {
            const line = chart.addSeries(LineSeries, {
              color: color(({basis: "muted", upper: "blue", lower: "pink"})[field] || metric.color),
              lineWidth: metric.key === "bands" ? 1 : 2,
              lineStyle: metric.dashed ? LineStyle.Dashed : LineStyle.Solid,
              priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
              priceFormat: {type: "custom", formatter: formatter(metric.pane), minMove: 0.01},
            }, index);
            line.setData(points);
          }
        }
      }
    }
    // Apply after creating the holder panes so they keep the default linear scale.
    candles.priceScale().applyOptions({
      mode: logarithmic ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal,
    });
    chart.panes()[0].setStretchFactor(paneKeys.length > 1 ? 3 : 1);
    // Give the real candles a shared calendar and permit bounded right-side
    // panning while leaving the future slots empty.
    const calendar = chart.addSeries(LineSeries, {visible: false}, 0);
    const last = rows.length - 1;
    const end = stamp(rows[last].candle.time);
    const future = Array.from({length: Math.ceil(rows.length * 2 / 3) + 2}, (_, i) =>
      ({time: isoDay(end + (i + 1) * (isWeek ? 7 : 1) * day)}));
    calendar.setData([...rows.map(row => ({time: row.candle.time})), ...future]);
    const scale = chart.timeScale();
    scale.subscribeVisibleLogicalRangeChange(range => {
      if (range) {
        const span = Math.min(Math.max(last, 1), range.to - range.from);
        const to = Math.max(span, Math.min(range.to, last + span * 2 / 3));
        if (Math.abs(to - range.to) > 0.001 || Math.abs(to - span - range.from) > 0.001) {
          scale.setVisibleLogicalRange({from: to - span, to});
        }
      }
      scheduleOverlay();
    });
    chart.subscribeCrosshairMove(event => {
      readout(event.time ? byTime.get(event.time) : rows.at(-1));
    });
    if (keepRange) scale.setVisibleLogicalRange(keepRange);
    else fit();
    readout(rows.at(-1));
    scheduleOverlay();
  }

  $("metric-menu").addEventListener("keydown", event => {
    if (event.key === "Escape") {
      $("metric-menu").open = false;
      $("metric-menu").querySelector("summary").focus();
    }
  });
  document.addEventListener("click", event => {
    if (!$("metric-menu").contains(event.target)) $("metric-menu").open = false;
  });
  $("interval").addEventListener("change", () => render());
  $("cycles").addEventListener("change", scheduleOverlay);
  $("ranges").addEventListener("click", event => {
    if (event.target.dataset.range) fit(event.target.dataset.range);
  });
  $("reset").addEventListener("click", () => {
    for (const pane of chart.panes()) pane.priceScale("right").applyOptions({autoScale: true});
    fit("365");
    readout(rows.at(-1));
  });
  $("log").addEventListener("click", event => {
    logarithmic = !logarithmic;
    event.currentTarget.setAttribute("aria-pressed", String(logarithmic));
    render(chart.timeScale().getVisibleLogicalRange());
  });
  $("theme").addEventListener("click", event => {
    const light = document.documentElement.dataset.theme === "dark";
    document.documentElement.dataset.theme = light ? "light" : "dark";
    event.currentTarget.textContent = light ? "Dark theme" : "Light theme";
    render(chart.timeScale().getVisibleLogicalRange());
  });
  const workspace = $("workspace");
  function updateMaximize() {
    $("maximize").textContent = document.fullscreenElement || workspace.classList.contains("expanded")
      ? "Restore chart" : "Maximize";
    scheduleOverlay();
  }
  $("maximize").addEventListener("click", async () => {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if (workspace.classList.contains("expanded")) workspace.classList.remove("expanded");
    else {
      try {
        await workspace.requestFullscreen();
      } catch {
        workspace.classList.add("expanded");
      }
    }
    updateMaximize();
  });
  document.addEventListener("fullscreenchange", updateMaximize);
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && workspace.classList.contains("expanded")) {
      workspace.classList.remove("expanded");
      updateMaximize();
    }
  });
  $("export").addEventListener("click", () => {
    const shot = chart.takeScreenshot();
    const image = document.createElement("canvas");
    image.width = shot.width;
    image.height = shot.height + 60;
    const context = image.getContext("2d");
    context.fillStyle = color("panel");
    context.fillRect(0, 0, image.width, 60);
    context.fillStyle = color("text");
    context.font = "bold 17px system-ui";
    const timeframe = $("interval").value === "weekly" ? "Weekly" : "Daily";
    context.fillText("Bitstamp BTC/USD · " + timeframe + " candles", 14, 25, image.width - 28);
    context.fillStyle = color("muted");
    context.font = "12px system-ui";
    context.fillText("Completed UTC sessions through " + rows.at(-1).date, 14, 45, image.width - 28);
    context.drawImage(shot, 0, 60);
    const download = document.createElement("a");
    download.href = image.toDataURL("image/png");
    download.download = "btc-market-" + rows.at(-1).date + ".png";
    download.click();
  });
  new ResizeObserver(scheduleOverlay).observe($("chart"));
  $("coverage").textContent = "Bitstamp BTC/USD · " + daily[0].date + " to " + daily.at(-1).date + " · Completed UTC sessions";
  checklist();
  render();
})();
