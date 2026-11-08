// Keep this file ASCII-only. Localized UI strings use C# Unicode escapes.
using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;

internal sealed class StartupWindow : Form
{
    private static readonly Color Rose = Color.FromArgb(153, 77, 110);
    private readonly Label status, footer;
    private readonly Label[] steps = new Label[3];
    private readonly RichTextBox output;
    private readonly Button close, openLog;
    private readonly LoadingBar loading;
    public event EventHandler CloseRequested;
    public event EventHandler OpenLogRequested;

    public StartupWindow()
    {
        Text = "Comfy\uacf5\ubc29 \u00b7 \uc2dc\uc791";
        ClientSize = new Size(620, 474);
        AutoScaleDimensions = new SizeF(96, 96);
        AutoScaleMode = AutoScaleMode.Dpi;
        StartPosition = FormStartPosition.CenterScreen;
        FormBorderStyle = FormBorderStyle.None;
        BackColor = Color.FromArgb(253, 251, 249);
        ForeColor = Color.FromArgb(63, 48, 58);
        Font = new Font("Malgun Gothic", 10);
        DoubleBuffered = true;
        ShowInTaskbar = true;
        var mark = AddLabel("\u2726", 28, 26, 52, 54, 28, FontStyle.Regular);
        mark.BackColor = Rose; mark.ForeColor = Color.White; mark.TextAlign = ContentAlignment.MiddleCenter;
        AddLabel("Comfy\uacf5\ubc29", 94, 24, 350, 40, 23, FontStyle.Bold);
        var subtitle = AddLabel("V6  \u00b7  CREATIVE WORKSPACE", 96, 67, 360, 20, 8, FontStyle.Regular);
        subtitle.ForeColor = Color.FromArgb(117, 97, 111);
        close = new Button { Text = "\u00d7", AccessibleName = "\uc2dc\uc791 \ucc3d \ub2eb\uae30", Location = new Point(568, 16), Size = new Size(36, 36),
            FlatStyle = FlatStyle.Flat, TabIndex = 1, ForeColor = Rose, Font = new Font("Segoe UI", 20) };
        close.FlatAppearance.BorderSize = 0;
        close.FlatAppearance.MouseOverBackColor = Color.FromArgb(245, 227, 236);
        close.Click += delegate { RequestClose(); };
        Controls.Add(close);
        status = AddLabel("\uc2e4\ud589 \ud658\uacbd\uc744 \ud655\uc778\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4\u2026", 28, 112, 560, 46, 12, FontStyle.Bold);
        loading = new LoadingBar { Location = new Point(28, 166), Size = new Size(564, 5), TabStop = false };
        Controls.Add(loading);
        string[] names = { "\uc2e4\ud589 \ud658\uacbd \ud655\uc778", "\ud654\uba74 \uc900\ube44", "\uc11c\ubc84 \uc5f0\uacb0" };
        for (int i = 0; i < steps.Length; i++)
            steps[i] = AddLabel((i + 1) + "  " + names[i], 28 + i * 190, 184, 185, 22, 9, FontStyle.Regular);
        AddLabel("\uc2e4\ud589 \ub85c\uadf8", 28, 223, 180, 24, 9, FontStyle.Bold);
        output = new RichTextBox { Location = new Point(28, 251), Size = new Size(564, 164), ReadOnly = true,
            BackColor = Color.FromArgb(36, 31, 39), ForeColor = Color.FromArgb(233, 222, 231), BorderStyle = BorderStyle.None,
            Font = new Font("Consolas", 9), DetectUrls = false, WordWrap = true, ScrollBars = RichTextBoxScrollBars.Vertical,
            AccessibleName = "\uc2e4\ud589 \ub85c\uadf8", TabIndex = 0 };
        Controls.Add(output);
        footer = AddLabel("\uc900\ube44\uac00 \ub05d\ub098\uba74 \uc774 \ucc3d\uc774 \ub2eb\ud788\uace0 \uc791\uc5c5 \uacf5\uac04\uc774 \uc5f4\ub9bd\ub2c8\ub2e4.", 28, 433, 430, 24, 9, FontStyle.Regular);
        footer.ForeColor = Color.FromArgb(117, 97, 111);
        openLog = new Button { Text = "\ub85c\uadf8 \ud30c\uc77c \uc5f4\uae30", Location = new Point(458, 428), Size = new Size(134, 30),
            FlatStyle = FlatStyle.Flat, ForeColor = Rose, Visible = false, TabIndex = 2 };
        openLog.FlatAppearance.BorderColor = Color.FromArgb(215, 200, 208);
        openLog.Click += delegate { if (OpenLogRequested != null) OpenLogRequested(this, EventArgs.Empty); };
        Controls.Add(openLog);
        SetStage(0, status.Text);
        FormClosing += delegate(object sender, FormClosingEventArgs e) { e.Cancel = true; RequestClose(); };
    }

    private Label AddLabel(string text, int x, int y, int width, int height, float size, FontStyle style)
    {
        var label = new Label { Text = text, Location = new Point(x, y), Size = new Size(width, height),
            Font = new Font("Malgun Gothic", size, style), BackColor = Color.Transparent };
        label.MouseDown += delegate(object sender, MouseEventArgs e) {
            if (e.Button == MouseButtons.Left) { ReleaseCapture(); SendMessage(Handle, 0xA1, (IntPtr)2, IntPtr.Zero); }
        };
        Controls.Add(label);
        return label;
    }

    // Localize only in the GUI; the startup protocol and console stay English.
    public void SetStartupStage(int current, string stage)
    {
        string text;
        switch (stage)
        {
            case "runtime": text = "\ucc98\uc74c \uc2e4\ud589\uc5d0 \ud544\uc694\ud55c \ud658\uacbd\uc744 \uc900\ube44\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4."; break;
            case "ui": text = "\ud654\uba74\uc744 \uc900\ube44\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4."; break;
            case "ui-dependencies": text = "\ud654\uba74 \uad6c\uc131 \uc694\uc18c\ub97c \uc900\ube44\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4."; break;
            case "ui-build": text = "\ubcc0\uacbd\ub41c \ud654\uba74\uc744 \uc801\uc6a9\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4."; break;
            case "server": text = "\uacf5\ubc29 \uc11c\ubc84\uc5d0 \uc5f0\uacb0\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4."; break;
            case "waiting": text = "\ub2e4\ub978 \ucc3d\uc5d0\uc11c \uacf5\ubc29\uc744 \uc900\ube44\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4. \uc7a0\uc2dc\ub9cc \uae30\ub2e4\ub824 \uc8fc\uc138\uc694."; break;
            case "ready": text = "\uc2e4\ud589 \uc911\uc778 \uacf5\ubc29\uc5d0 \uc5f0\uacb0\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4."; break;
            default: text = "\uc2e4\ud589 \ud658\uacbd\uc744 \ud655\uc778\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4."; break;
        }
        SetStage(current, text);
    }

    public void SetStage(int current, string text)
    {
        status.Text = text;
        for (int i = 0; i < steps.Length; i++)
            steps[i].ForeColor = i <= current ? Rose : Color.FromArgb(162, 143, 155);
        if (current == 3) loading.Complete();
    }

    public void AppendLog(string line, bool error)
    {
        if (line.StartsWith("COMFY_STARTUP|", StringComparison.Ordinal))
        {
            int divider = line.IndexOf('|', "COMFY_STARTUP|".Length);
            if (divider >= 0) line = line.Substring(divider + 1);
        }
        bool follow = !output.Focused || output.SelectionStart + output.SelectionLength >= output.TextLength;
        if (output.TextLength > 180000) { output.Select(0, output.TextLength - 120000); output.SelectedText = ""; }
        output.SelectionStart = output.TextLength;
        output.SelectionLength = 0;
        output.SelectionColor = error ? Color.FromArgb(255, 167, 180) : output.ForeColor;
        output.AppendText(line + Environment.NewLine);
        if (follow) output.ScrollToCaret();
    }

    public void ShowFailure(string message, string logPath)
    {
        Text = "Comfy\uacf5\ubc29 \u00b7 \uc2dc\uc791 \uc624\ub958";
        loading.Stop();
        status.Text = message;
        status.ForeColor = Rose;
        footer.Text = "\uc624\ub958 \ub0b4\uc6a9\uc744 \ud655\uc778\ud55c \ub4a4 X\ub85c \ub2eb\uc544 \uc8fc\uc138\uc694.";
        openLog.Visible = !String.IsNullOrEmpty(logPath);
        close.Enabled = true;
        close.AccessibleName = "\uc624\ub958 \ucc3d \ub2eb\uae30";
    }

    public void DisableClose() { close.Enabled = false; }
    private void RequestClose() { if (close.Enabled && CloseRequested != null) CloseRequested(this, EventArgs.Empty); }
    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        using (var border = new Pen(Color.FromArgb(215, 200, 208)))
            e.Graphics.DrawRectangle(border, 0, 0, ClientSize.Width - 1, ClientSize.Height - 1);
    }
    protected override CreateParams CreateParams { get { var value = base.CreateParams; value.ClassStyle |= 0x20000; return value; } }
    [DllImport("user32.dll")] private static extern bool ReleaseCapture();
    [DllImport("user32.dll")] private static extern IntPtr SendMessage(IntPtr window, int message, IntPtr wParam, IntPtr lParam);
}

internal sealed class LoadingBar : Control
{
    private readonly Timer animation = new Timer { Interval = 25 };
    private int position;
    private bool completed;
    public LoadingBar()
    {
        DoubleBuffered = true;
        animation.Tick += delegate { position = (position + 5) % Math.Max(1, Width + 100); Invalidate(); };
        animation.Start();
    }
    public void Complete() { completed = true; animation.Stop(); Invalidate(); }
    public void Stop() { animation.Stop(); }
    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.Clear(Color.FromArgb(239, 226, 233));
        using (var brush = new SolidBrush(Color.FromArgb(153, 77, 110)))
            e.Graphics.FillRectangle(brush, completed ? 0 : position - 100, 0, completed ? Width : 100, Height);
    }
    protected override void Dispose(bool disposing) { if (disposing) animation.Dispose(); base.Dispose(disposing); }
}
