// Keep this file ASCII-only. Localized UI strings use C# Unicode escapes.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class SettingsLauncher
{
    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        bool noBrowser = Array.IndexOf(args, "--no-browser") >= 0;
        if (WorkshopApplication.TryOpenRunning(noBrowser)) return Environment.ExitCode;
        using (var application = new WorkshopApplication(noBrowser))
            Application.Run(application);
        return Environment.ExitCode;
    }
}

internal sealed class WorkshopApplication : ApplicationContext
{
    private const string SettingsUrl = "http://127.0.0.1:8201/setting/";
    private const string StatusPrefix = "COMFY_STARTUP|";
    private readonly StartupWindow window = new StartupWindow();
    private readonly System.Windows.Forms.Timer poll = new System.Windows.Forms.Timer();
    private readonly object logLock = new object();
    private readonly bool noBrowser;
    private Process process;
    private StreamWriter log;
    private string logPath, startupError;
    private volatile bool ready;
    private bool failed, closing, waitingForOther, pollEnabled, processExited;
    private int probing;

    public static bool TryOpenRunning(bool noBrowser)
    {
        try { if (!RunningServer(700)) return false; }
        catch (Exception) { return false; }
        try { if (!noBrowser) Process.Start(new ProcessStartInfo(SettingsUrl) { UseShellExecute = true }); }
        catch (Exception)
        {
            Environment.ExitCode = 1;
            MessageBox.Show("\uacf5\ubc29\uc740 \uc2e4\ud589 \uc911\uc774\uc9c0\ub9cc \ube0c\ub77c\uc6b0\uc800\ub97c \uc5f4\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.\n" + SettingsUrl + " \uc8fc\uc18c\ub85c \uc811\uc18d\ud574 \uc8fc\uc138\uc694.",
                "Comfy\uacf5\ubc29", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
        return true;
    }

    private static bool RunningServer(int timeout)
    {
        var request = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:8201/api/setting/status");
        request.Proxy = null;
        request.Timeout = timeout;
        request.ReadWriteTimeout = timeout;
        using (var response = request.GetResponse())
        using (var reader = new StreamReader(response.GetResponseStream(), Encoding.UTF8))
        {
            var status = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(reader.ReadToEnd());
            object name, mode;
            return status != null && status.TryGetValue("name", out name) && status.TryGetValue("mode", out mode)
                && Convert.ToString(name) == "Comfy\uacf5\ubc29" && Convert.ToString(mode) == "settings";
        }
    }

    public WorkshopApplication(bool noBrowser)
    {
        this.noBrowser = noBrowser;
        window.CloseRequested += CloseRequested;
        window.OpenLogRequested += delegate {
            try { Process.Start(new ProcessStartInfo(logPath) { UseShellExecute = true }); }
            catch (Exception error) { window.AppendLog(error.Message, true); }
        };
        window.Shown += delegate { Start(); };
        poll.Interval = 500;
        poll.Tick += delegate { Probe(); };
        window.Show();
    }

    private void Start()
    {
        try
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string directory = Path.Combine(root, "data", "logs");
            Directory.CreateDirectory(directory);
            logPath = Path.Combine(directory, "launcher-" + DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture)
                + "-" + Process.GetCurrentProcess().Id + ".log");
            log = new StreamWriter(new FileStream(logPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read), new UTF8Encoding(false));
            log.AutoFlush = true;
            string script = Path.Combine(root, "src", "run_settings.ps1");
            if (!File.Exists(script)) throw new FileNotFoundException("Place the launcher next to the project's src folder.", script);
            string powershell = Path.Combine(Environment.SystemDirectory, "WindowsPowerShell", "v1.0", "powershell.exe");
            // Configure encoding before parsing the script, so even its syntax
            // errors are readable. Single-quote the literal path before encoding.
            string command = "$ProgressPreference = 'SilentlyContinue'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
                + "$OutputEncoding = [Console]::OutputEncoding; $ErrorActionPreference = 'Stop'; "
                + "try { & '" + script.Replace("'", "''") + "' -LauncherMode; exit $LASTEXITCODE } "
                + "catch { [Console]::Error.WriteLine(($_ | Out-String)); exit 1 }";
            var start = new ProcessStartInfo(powershell,
                "-NoLogo -NoProfile -NonInteractive -OutputFormat Text -WindowStyle Hidden -ExecutionPolicy Bypass -EncodedCommand "
                + Convert.ToBase64String(Encoding.Unicode.GetBytes(command)));
            start.WorkingDirectory = root;
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.WindowStyle = ProcessWindowStyle.Hidden;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            start.RedirectStandardInput = true;
            start.StandardOutputEncoding = new UTF8Encoding(false);
            start.StandardErrorEncoding = new UTF8Encoding(false);
            start.EnvironmentVariables["PYTHONUTF8"] = "1";
            start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8:backslashreplace";
            process = new Process { StartInfo = start, EnableRaisingEvents = true };
            process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { Output(e.Data, false); };
            process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { Output(e.Data, true); };
            process.Exited += delegate {
                // Drain redirected streams before presenting the final error.
                process.WaitForExit();
                int code = process.ExitCode;
                OnUi(delegate {
                    processExited = true;
                    if (closing) return;
                    if (code != 0) Fail(startupError ?? "\uc2dc\uc791 \uc911 \uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4. \uc544\ub798 \uc2e4\ud589 \ub85c\uadf8\ub97c \ud655\uc778\ud574 \uc8fc\uc138\uc694.");
                    else if (ready) Finish();
                    else if (!pollEnabled) Fail("\uc11c\ubc84\ub97c \uc2dc\uc791\ud558\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4. \uc544\ub798 \uc2e4\ud589 \ub85c\uadf8\ub97c \ud655\uc778\ud574 \uc8fc\uc138\uc694.");
                    else Probe();
                });
            };
            process.Start();
            process.StandardInput.Close();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            poll.Start();
        }
        catch (Exception error) { Output(error.ToString(), true); Fail(error.Message); }
    }

    private void Output(string line, bool isError)
    {
        if (line == null) return;
        lock (logLock)
        {
            // The server owns its runtime logs; this file keeps startup output.
            if (log != null && (!ready || isError))
                try { log.WriteLine(line); } catch (IOException) { }
        }
        if (ready && !isError) return;
        OnUi(delegate {
            if (!ready || isError) window.AppendLog(line, isError);
            if (isError || !line.StartsWith(StatusPrefix, StringComparison.Ordinal)) return;
            string[] fields = line.Substring(StatusPrefix.Length).Split(new char[] { '|' }, 2);
            if (fields.Length != 2) return;
            string stage = fields[0];
            if (stage == "error") { startupError = fields[1]; return; }
            int step = stage == "ui" || stage.StartsWith("ui-", StringComparison.Ordinal) ? 1
                : stage == "server" || stage == "waiting" || stage == "ready" ? 2 : 0;
            window.SetStartupStage(step, stage);
            if (stage == "waiting") waitingForOther = true;
            if (step == 2) { pollEnabled = true; Probe(); }
        });
    }

    private void Probe()
    {
        if (!pollEnabled || ready || failed || closing || Interlocked.CompareExchange(ref probing, 1, 0) != 0) return;
        ThreadPool.QueueUserWorkItem(delegate {
            bool available = false;
            string error = null;
            try
            {
                available = RunningServer(1500);
                if (!available) error = "8201 \ud3ec\ud2b8\ub97c \ub2e4\ub978 \ud504\ub85c\uadf8\ub7a8\uc774 \uc0ac\uc6a9\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4.";
            }
            catch (WebException) { }
            catch (Exception exception) { error = exception.Message; }
            finally { Interlocked.Exchange(ref probing, 0); }
            OnUi(delegate {
                if (ready || failed || closing) return;
                if (available) OpenWorkspace();
                else if (error != null) Fail(error);
                else if (processExited && !waitingForOther) Fail("\uc11c\ubc84\uac00 \uc900\ube44\ub418\uae30 \uc804\uc5d0 \uc885\ub8cc\ub418\uc5c8\uc2b5\ub2c8\ub2e4. \uc544\ub798 \uc2e4\ud589 \ub85c\uadf8\ub97c \ud655\uc778\ud574 \uc8fc\uc138\uc694.");
            });
        });
    }

    private void OpenWorkspace()
    {
        ready = true;
        poll.Stop();
        window.SetStage(3, "\uc900\ube44\uac00 \ub05d\ub0ac\uc2b5\ub2c8\ub2e4. \uc791\uc5c5 \uacf5\uac04\uc744 \uc5fd\ub2c8\ub2e4.");
        try { if (!noBrowser) Process.Start(new ProcessStartInfo(SettingsUrl) { UseShellExecute = true }); }
        catch (Exception error)
        {
            Output(error.ToString(), true);
            window.ShowFailure("\ube0c\ub77c\uc6b0\uc800\ub97c \uc5f4\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4. " + SettingsUrl + " \uc8fc\uc18c\ub85c \uc811\uc18d\ud574 \uc8fc\uc138\uc694.", logPath);
            return;
        }
        // Keep draining the hidden server's pipes until it exits, without a window.
        var dismiss = new System.Windows.Forms.Timer { Interval = 500 };
        dismiss.Tick += delegate {
            dismiss.Stop(); dismiss.Dispose();
            if (closing || failed) return;
            window.Hide();
            if (processExited) Finish();
        };
        dismiss.Start();
    }

    private void Fail(string message)
    {
        if (closing) return;
        failed = true;
        poll.Stop();
        Environment.ExitCode = 1;
        window.ShowFailure(message, logPath);
        if (!window.Visible) window.Show();
    }

    private void CloseRequested(object sender, EventArgs e)
    {
        if (closing) return;
        if (ready && process != null && !process.HasExited) { window.Hide(); return; }
        closing = true;
        poll.Stop();
        window.SetStage(0, "\uc2dc\uc791 \uc900\ube44\ub97c \uc911\ub2e8\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4\u2026");
        window.DisableClose();
        ThreadPool.QueueUserWorkItem(delegate {
            try
            {
                if (process != null && !process.HasExited)
                {
                    // Only terminate the process tree owned by this launch.
                    var stop = new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "taskkill.exe"),
                        "/PID " + process.Id + " /T /F") { UseShellExecute = false, CreateNoWindow = true, WindowStyle = ProcessWindowStyle.Hidden };
                    using (Process command = Process.Start(stop)) command.WaitForExit();
                    process.WaitForExit();
                }
            }
            catch (Exception error) { Output(error.ToString(), true); }
            OnUi(Finish);
        });
    }

    private void OnUi(Action action)
    {
        if (window.IsDisposed || !window.IsHandleCreated) return;
        try { window.BeginInvoke(action); } catch (InvalidOperationException) { }
    }

    private void Finish()
    {
        closing = true;
        poll.Stop();
        lock (logLock) { if (log != null) { log.Dispose(); log = null; } }
        window.Dispose();
        ExitThread();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            poll.Dispose();
            if (process != null) process.Dispose();
            lock (logLock) { if (log != null) { log.Dispose(); log = null; } }
            window.Dispose();
        }
        base.Dispose(disposing);
    }
}
