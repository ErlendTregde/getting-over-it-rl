using BepInEx;
using HarmonyLib;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using UnityEngine.SceneManagement;

[BepInPlugin("local.goi.bridge", "GOI Bridge", "0.57.0")]
public class Bridge : BaseUnityPlugin
{
    public const int PORT = 9955;
    const BindingFlags F = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    static readonly CultureInfo INV = CultureInfo.InvariantCulture;

    public static Bridge I;

    // ---- player refs ----
    public MonoBehaviour player;
    Rigidbody2D root, tipRB, hingeRB, cursorRB;
    HingeJoint2D hj;
    SliderJoint2D sj;
    FieldInfo fMouseInput, fSensitivity;
    MethodInfo miFixedUpdate;
    float dt = 1f / 120f;

    // ---- progress: the game's own metric ----
    // Foddy authored a CurvySpline up the mountain; ProgressMeter projects the
    // player onto it and reports arc length. It does that in LateUpdate, which
    // does not run during manual physics ticks, so we query the spline directly.
    object spline;
    MethodInfo miNearestTF, miTFToDistance, miInterpByDist, miToWorld, miDistToTF;
    PropertyInfo piSplineLength, piSplineCount;

    // Nearest-point projection searches the whole spline by default, and where
    // the mountain folds back on itself it will happily snap to a stretch a
    // hundred units away in arc. Constraining the search to segments near where
    // we already were removes that -- a single tick cannot move the pot far.
    // Negative winBack means search everything (needed while settling a
    // teleported pot, which legitimately falls a long way in one command).
    float lastArc = -1f;
    float winBack = -1f, winFwd = -1f;

    // Name of the scene the player was found in, so a run that trips the ending
    // sequence can be put back into the game.
    string gameScene = "";

    // Mutable controller state (oldAngle, oldHammerPos, mouseVelocityAverage,
    // ...). Restoring rigidbodies alone leaves these stale, which makes a
    // restored state diverge from the one that was saved.
    FieldInfo[] pcFields = new FieldInfo[0];
    FieldInfo fiInputEnabled;

    // Unity interpolates a rigidbody's TRANSFORM between physics states using
    // real elapsed time. Under simulationMode = Script that clock no longer
    // matches when physics actually advances, so every transform is drawn
    // chasing a stale target. The bodies are in the right places -- rb.position
    // is correct -- but a chain of them renders visibly compressed, which is
    // why the hammer looks about half length whenever a script is driving.
    readonly Dictionary<Rigidbody2D, RigidbodyInterpolation2D> interpWas =
        new Dictionary<Rigidbody2D, RigidbodyInterpolation2D>();

    // Controller fields that are NOT physics. Capturing every float and bool on
    // PlayerControl swept these up, and restoring them made a checkpoint carry
    // "input is switched off" into the future: after any load the human's mouse
    // did nothing and the hammer simply hung, which reads on screen as the
    // hammer having got shorter. Blobs are decoded by NAME, so dropping these
    // leaves existing checkpoint files working -- the stale entries just no
    // longer match anything.
    static readonly HashSet<string> NotState = new HashSet<string> {
        "input_enabled", "inputsToSkip", "skipfirstMoveInput", "pauseInputTimer",
        "menuPause", "loadedFromSave", "loadFinished", "numWins",
        // settings, not state: restoring these would silently reset the
        // player's own sensitivity from a months-old snapshot
        "mouseSensitivity", "deadzone", "trackpad", "mobileScreenDPIAdjust",
        "angleEpsilon", "posEpsilon",
    };

    // ---- control state ----
    public bool lockstep = false;
    public bool agentControlled = false;
    public static bool inManualTick = false;
    Vector2 injected = Vector2.zero;

    // Human demonstration capture. Recorded HERE, at the physics rate, rather
    // than polled over the socket: the whole point is the exact shape of the
    // mouse movement, and a 30Hz sample of a 120Hz input would smooth away the
    // very thing being looked for.
    bool demoRec = false;
    string demoKey = null;               // what F restores
    readonly List<float[]> demo = new List<float[]>();
    int demoAttempt = 0;
    const int DEMO_MAX = 120 * 60 * 12;  // twelve minutes, then it stops growing

    // The observation/action readout: what the agent actually senses and does,
    // drawn over the game. Everything it shows is read live from the same
    // bodies Obs() reads, so it costs no bridge traffic and cannot disagree
    // with what the policy was fed.
    bool obsPanel = false;
    float obsPadScale = 13.8f;      // the action scale the trainer injects at

    // Following from the client means a camfix per step: 30 updates a second
    // against a 60+ fps render, which reads as stutter no matter how the
    // client is paced. Done here it is smooth by construction and free.
    bool camFollow = false;
    float camFollowEase = 0.18f;
    Vector2 camSmooth = Vector2.zero;
    static Texture2D pxFill, pxFrame, pxBack, pxMark, pxHot;
    static Texture2D cRing, cDisc, cDot;
    public Vector2 Injected { get { return injected; } }

    // Largest |axis| seen from a real mouse, so the action scale can be matched
    // to the range a human actually produces instead of guessed.
    public float humanAxisMax = 0f;

    // The action a HUMAN is producing, signed, per axis. humanAxisMax only ever
    // held a magnitude, which is useless for a pad that has to show direction.
    public Vector2 humanAxis = Vector2.zero;
    public void NoteHumanAxis(string name, float v)
    {
        if (name == "mouseX") humanAxis.x = v; else humanAxis.y = v;
    }

    // ---- terrain perception ----
    // The observation was entirely proprioceptive: everything about the pot and
    // the hammer, nothing about the mountain. A policy could only memorise what
    // to do at each arc position, with no transfer between sections. These rays
    // are what let it learn what a ledge looks like instead.
    public const int RAYS_BODY = 16;
    public const int RAYS_TIP = 8;
    const float RANGE_BODY = 14f;
    const float RANGE_TIP = 8f;
    int terrainMask = ~0;

    // ---- capture: drawing what the agent perceives ----
    // The rays are already computed for the observation; this draws the same
    // fan in the world so a recording shows what the policy is actually working
    // from, rather than a pot moving for no visible reason.
    GameObject vizRoot;
    LineRenderer[] vizLines;
    bool viz = false, hud = false;
    // Which fans to draw: bit 1 = the 16 around the pot, bit 2 = the 8 at the
    // hammer head. Showing one at a time is how a shot explains what each set
    // is actually for.
    int vizMask = 3;

    // World-space markers, so the checkpoint ladder can be seen on the mountain
    // instead of read as a list of numbers.
    class Marker { public Vector2 p; public int kind; public string key; }
    readonly List<Marker> markers = new List<Marker>();

    // "cp001 16.4 -> gx000 20.9", set by the trainer so the corner readout says
    // which rung this episode started from and which one it is aiming at.
    string routeLabel = "";
    GameObject markerRoot;

    // Recording aids. `hudtext` splits the corner readout from the marker
    // labels: footage wants clean geometry, not a debug overlay burned into it.
    bool hudText = true;
    // `isolate` blacks out the world and leaves only the pot and our own lines,
    // so a shot can show what the agent perceives with nothing else competing.
    bool isolated = false;
    readonly List<Renderer> hidden = new List<Renderer>();
    CameraClearFlags isoClear = CameraClearFlags.Skybox;
    Color isoBg = Color.black;
    // Ghosts: many positions drawn at once, updated every frame WITHOUT
    // rebuilding GameObjects. Rebuilding a marker set per frame is fine once a
    // second and unwatchable at 60fps.
    GameObject ghostRoot;
    readonly List<LineRenderer> ghostPool = new List<LineRenderer>();

    // Real players for the montage. The game has one pot, so every model
    // climbing at once has to be visual-only copies of the player driven by
    // trajectories recorded earlier -- the same trick a Trackmania ghost is.
    // Physics, colliders, joints, cameras and lights are stripped from a clone;
    // what is left is the mesh hierarchy, posed transform by transform.
    GameObject rigRoot;
    readonly List<Transform[]> rigs = new List<Transform[]>();

    // Pose tracks held plugin-side. Streaming poses every frame costs 5 KB per
    // model per frame, so a hundred models is 16 MB/s of text and playback
    // collapses to a slideshow. Upload each track once and a frame becomes
    // "ghostplay 412" -- flat cost, however many models are on screen.
    readonly List<float[][]> ghostTracks = new List<float[][]>();

    // Frames to wait before a clone enters the shot. A model that is meant
    // to arrive late must be ABSENT until it does -- parking it on the start
    // line where it stands motionless for ten seconds reads as broken, not as
    // dramatic. Visibility is cached because toggling renderers on a hundred
    // rigs every frame is not free.
    readonly List<int> ghostDelay = new List<int>();
    readonly List<bool> ghostShown = new List<bool>();

    // Which transforms a stored frame carries, as indices into poseOrder.
    // Most of the 144 never move relative to their parent; re-sending them
    // thirty times a second is the whole budget at high model counts.
    int[] poseMask = null;
    Transform[] poseOrder = new Transform[0];

    // A take starts from the game window, not the terminal: whoever is
    // recording has the game focused and should not have to alt-tab to a
    // console to trigger the shot they are filming.
    bool cued = false;

    // The real player has to disappear for a montage. Physics is frozen while
    // the clones replay, so it stands motionless in shot as an extra figure
    // that never moves -- which is the one thing in frame that looks broken.
    readonly List<Renderer> playerHidden = new List<Renderer>();

    // Flying is handled here rather than in the client, because the keystrokes
    // have to go to the window you are looking at. Typing into a terminal while
    // watching the game meant every key hit the plugin's own hotkeys instead.
    bool flymode = false;
    int markerIndex = 0;
    const float FLY_SPEED = 22f;
    // Edits are requested in the game window and collected by the client, which
    // is the side that owns the checkpoint file.
    bool pendingAdd = false, pendingDelete = false;

    // Hand the controls back to the human. Flying places the pot but leaves the
    // hammer wherever it falls, and a checkpoint is only as good as its hammer
    // pose -- the agent inherits it exactly. Playing the move yourself and
    // saving mid-swing produces poses that are known to work, because someone
    // just made them work.
    bool manual = false;

    // A fixed wide shot: training resets to a different rung several times a
    // second, so a following camera is unwatchable. Locking it over the whole
    // section being trained shows every attempt in one frame.
    Behaviour camControl;
    bool camLocked = false;
    Vector3 camHome, camTarget;
    // Live camera height while locked, so scrolling can change it without the
    // trainer having to re-send camfix (which it only does once, at startup).
    float camHeight = 30f;
    Vector3 dragOrigin;
    bool dragging = false;
    float camHomeOrtho;

    // ---- cameras ----
    class CamState { public Camera cam; public int mask; public CameraClearFlags clear; public bool enabled; }
    readonly List<CamState> cams = new List<CamState>();

    // Frame rate, saved so the game gets its own back.
    bool fpsSaved = false;
    int vsyncWas = 1, fpsWas = -1;
    float fpsAvg = 60f;

    // What the anti-aliasing mode was before we touched it, per layer, so the
    // game gets its own look back.
    readonly Dictionary<MonoBehaviour, object> aaWas =
        new Dictionary<MonoBehaviour, object>();
    readonly List<object> blurOff = new List<object>();

    // ---- save states ----
    class Snap { public Rigidbody2D rb; public string name; public Vector2 p, v; public float r, w;
                 // Rigidbody2D has no depth, but the transform does, and the
                 // renderer needs it. Optional so older blobs still load.
                 public float? z; }
    // Gravity belongs in the snapshot: it is global mutable state that the game
    // changes via triggers (GravityControl weakens it for the space section near
    // the top). Restoring bodies without it means a state recorded under one
    // gravity is replayed under whatever the last trigger happened to set --
    // which silently made checkpoints in open sky look stable.
    class State
    {
        public List<Snap> bodies = new List<Snap>();
        public object[] ctrl;
        public Vector2 gravity;
        public bool hasGravity;
    }
    readonly Dictionary<string, State> saves = new Dictionary<string, State>();

    // ---- net plumbing ----
    Thread netThread;
    volatile bool running = true;
    string pendingCmd, pendingResult;
    readonly AutoResetEvent reqReady = new AutoResetEvent(false);
    readonly AutoResetEvent resReady = new AutoResetEvent(false);

    Harmony harmony;
    bool patched = false;
    bool axisPatched = false;

    void Awake()
    {
        I = this;
        Application.runInBackground = true;
        harmony = new Harmony("local.goi.bridge");
        netThread = new Thread(NetLoop) { IsBackground = true, Name = "goi-bridge-net" };
        netThread.Start();
        SceneManager.sceneLoaded += OnSceneLoaded;
        Logger.LogInfo($"bridge 0.57.0 listening on 127.0.0.1:{PORT}");
        Logger.LogInfo("hotkeys: I=discover  O=state  J=save  N=load  K=freeze  L=step1s  P=speed");
    }

    void OnSceneLoaded(Scene sc, LoadSceneMode mode)
    {
        // Every Rigidbody2D a save state points at has just been destroyed, and
        // so has the player. Drop it all; the next command re-discovers, and the
        // client re-uploads its checkpoints.
        Logger.LogWarning($"scene '{sc.name}' loaded ({mode}) — player, spline and "
                          + $"{saves.Count} save states invalidated");
        player = null; spline = null;
        saves.Clear(); cams.Clear();
        lastArc = -1f;
        EnsureSimulationMode();     // a scene load resets it; lockstep must survive
    }

    void OnDestroy()
    {
        running = false;
        try { SceneManager.sceneLoaded -= OnSceneLoaded; } catch { }
        try { harmony?.UnpatchSelf(); } catch { }
    }

    // =====================================================================
    // main thread pump — executes queued commands inside Unity's Update
    // =====================================================================
    void Update()
    {
        // Smoothed, so `fps` reports something stable enough to act on.
        if (Time.unscaledDeltaTime > 1e-5f)
            fpsAvg += (1f / Time.unscaledDeltaTime - fpsAvg) * 0.05f;
        // The game clears this on its own during startup, so re-assert it. Safe
        // to do from Update: as long as it is true when focus is lost, Update
        // keeps being called and can keep it true.
        if (!Application.runInBackground) Application.runInBackground = true;

        Hotkeys();
        FlyControls();

        // In lockstep the world is frozen between commands, so there is no UI to
        // keep responsive — spend most of the frame servicing step commands.
        float budgetEnd = Time.realtimeSinceStartup + (lockstep ? 0.25f : 0.012f);
        while (reqReady.WaitOne(1))
        {
            string cmd = pendingCmd;
            string res;
            try { res = Execute(cmd); }
            catch (Exception e) { res = "err " + e.Message.Replace('\n', ' '); }
            pendingResult = res;
            resReady.Set();
            if (Time.realtimeSinceStartup > budgetEnd) break;
        }
    }

    void FlyControls()
    {
        if (!flymode || player == null || root == null) return;

        if (Input.GetKeyDown(KeyCode.M))
        {
            manual = !manual;
            SetLockstep(!manual);      // manual needs the world running
            agentControlled = !manual; // and the real mouse driving the hammer
            Logger.LogInfo(manual
                ? "MANUAL: play with the mouse; ENTER saves this exact pose"
                : "FROZEN: arrows fly, 8 = prev / 9 = next, ENTER saves");
        }
        if (Input.GetKeyDown(KeyCode.Return)) pendingAdd = true;
        if (Input.GetKeyDown(KeyCode.Delete) ||
            Input.GetKeyDown(KeyCode.Backspace)) pendingDelete = true;
        if (manual) return;            // mouse belongs to the hammer now

        float mult = (Input.GetKey(KeyCode.LeftShift) ||
                      Input.GetKey(KeyCode.RightShift)) ? 4f : 1f;
        Vector2 d = Vector2.zero;
        if (Input.GetKey(KeyCode.UpArrow)) d.y += 1f;
        if (Input.GetKey(KeyCode.DownArrow)) d.y -= 1f;
        if (Input.GetKey(KeyCode.LeftArrow)) d.x -= 1f;
        if (Input.GetKey(KeyCode.RightArrow)) d.x += 1f;
        if (d != Vector2.zero)
            Teleport(root.position + d.normalized * FLY_SPEED * mult *
                     Time.unscaledDeltaTime);

        if (markers.Count > 0)
        {
            int step = (Input.GetKeyDown(KeyCode.Alpha9) ||
                        Input.GetKeyDown(KeyCode.Keypad9) ||
                        Input.GetKeyDown(KeyCode.PageUp) ||
                        Input.GetKeyDown(KeyCode.RightBracket)) ? 1
                     : (Input.GetKeyDown(KeyCode.Alpha8) ||
                        Input.GetKeyDown(KeyCode.Keypad8) ||
                        Input.GetKeyDown(KeyCode.PageDown) ||
                        Input.GetKeyDown(KeyCode.LeftBracket)) ? -1 : 0;
            if (step != 0)
            {
                markerIndex = (markerIndex + step + markers.Count) % markers.Count;
                var m = markers[markerIndex];
                // Restore the actual saved state where there is one. Teleporting
                // to the position instead shows the pot in whatever pose it is
                // carrying, which is not what an episode starting here would see.
                if (m.key != null && m.key.Length > 0 && Load(m.key)) lastArc = -1f;
                else Teleport(m.p);
            }
        }

        // let go and watch it fall
        if (Input.GetKeyDown(KeyCode.Space)) Tick(0f, 0f, 120);
    }

    void Hotkeys()
    {
        // Deliberately outside every guard below -- the cue has to work while
        // the agent is driving, which is exactly when a take is being filmed.
        // Flymode owns Return for placing checkpoints, so stay off it there.
        if (!flymode && Input.GetKeyDown(KeyCode.Return)) cued = true;

        // The letter hotkeys would fire under the fingers of anyone flying.
        if (flymode) return;
        // ... and while the agent is driving they are worse than useless: K
        // toggles lockstep, N teleports the pot to an old save, L advances the
        // sim behind the trainer's back, P changes timeScale. Watching a run
        // must be read-only, so that a stray keypress cannot corrupt it.
        if (agentControlled) return;
        // F: back to the demo start. Marks the recording so attempts can be
        // told apart, rather than reading as one impossible continuous run.
        if (Input.GetKeyDown(KeyCode.F) && demoKey != null)
        {
            if (Load(demoKey))
            {
                demoAttempt++;
                if (demoRec && demo.Count < DEMO_MAX)
                    demo.Add(new float[] { -1f, demoAttempt, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0 });
                Logger.LogInfo("demo: attempt " + demoAttempt);
            }
        }
        if (Input.GetKeyDown(KeyCode.I)) { Discover(); Logger.LogInfo(player != null ? "discovered" : "not found"); }
        if (Input.GetKeyDown(KeyCode.O)) Logger.LogInfo(Obs());
        if (Input.GetKeyDown(KeyCode.J)) { Save("hotkey"); Logger.LogInfo("saved"); }
        if (Input.GetKeyDown(KeyCode.N)) { Logger.LogInfo(Load("hotkey") ? "restored" : "no save"); }
        if (Input.GetKeyDown(KeyCode.K)) { SetLockstep(!lockstep); Logger.LogInfo("lockstep " + lockstep); }
        if (Input.GetKeyDown(KeyCode.L) && lockstep) { Tick(0, 0, 120); Logger.LogInfo("stepped 1s"); }
        if (Input.GetKeyDown(KeyCode.P))
        {
            Time.timeScale = Time.timeScale > 1.5f ? 1f : 5f;
            Logger.LogInfo("timeScale " + Time.timeScale);
        }
    }

    // =====================================================================
    // command dispatch
    // =====================================================================
    string Execute(string line)
    {
        var p = line.Split(' ');
        if (player == null && p[0] != "hello") Discover();
        switch (p[0])
        {
            case "hello":
                return "ok bridge 0.57.0";

            case "discover":
                Discover();
                return player != null ? "ok " + Obs() : "err no_player";

            case "obs":
                if (player == null) return "err no_player";
                return "ok " + Obs();

            case "step":
                {
                    if (player == null) return "err no_player";
                    float dx = Flt(p[1]), dy = Flt(p[2]);
                    int n = p.Length > 3 ? int.Parse(p[3], INV) : 1;
                    Tick(dx, dy, n);
                    return "ok " + Obs();
                }

            case "save":
                if (player == null) return "err no_player";
                Save(p.Length > 1 ? p[1] : "default");
                return "ok";

            case "load":
                {
                    string k = p.Length > 1 ? p[1] : "default";
                    if (!Load(k)) return "err no_save";
                    // Deliberately ignore any arc the caller offers. Seeding the
                    // search from a claimed value lets a wrong arc anchor the
                    // window around the wrong stretch of spline, where every
                    // subsequent reading agrees with it and the error can never
                    // be detected. -1 forces one global re-projection, which is
                    // the game's own definition of progress and cannot drift.
                    lastArc = -1f;
                    return "ok " + Obs();
                }

            case "savekeys":
                return "ok " + string.Join(",", new List<string>(saves.Keys).ToArray());

            case "lockstep":
                SetLockstep(p[1] == "1");
                return "ok";

            case "agent":
                agentControlled = p[1] == "1";
                // Giving the world back means the human's input has to work
                // again. The game leaves input_enabled false in plenty of
                // situations, and a checkpoint used to carry that state
                // forward, so "agent 0" alone was not enough.
                if (!agentControlled) SetInputEnabled(true);
                return "ok agent " + agentControlled;

            case "tree":
                // Every transform under the player, with the scales that decide
                // how big things are DRAWN. rb.position and transform.position
                // agreeing proves the simulation is right; it says nothing
                // about what a renderer is stretched to.
                {
                    if (player == null) return "err no_player";
                    var sb4 = new StringBuilder();
                    foreach (var tr in player.transform.root
                                             .GetComponentsInChildren<Transform>(true))
                    {
                        var r = tr.GetComponent<Renderer>();
                        var ls = tr.localScale;
                        var ws = tr.lossyScale;
                        if (sb4.Length > 0) sb4.Append(';');
                        var wp = tr.position;
                        sb4.Append(tr.name).Append(',')
                           .Append(ls.x.ToString("F3", INV)).Append(',')
                           .Append(ls.y.ToString("F3", INV)).Append(',')
                           .Append(ws.x.ToString("F3", INV)).Append(',')
                           .Append(ws.y.ToString("F3", INV)).Append(',')
                           .Append(wp.z.ToString("F4", INV)).Append(',')
                           .Append(r == null ? "-" : (r.enabled ? r.GetType().Name : "off"))
                           .Append(',')
                           .Append(tr.gameObject.activeInHierarchy ? "on" : "OFF")
                           .Append(',')
                           .Append(wp.x.ToString("F3", INV)).Append(',')
                           .Append(wp.y.ToString("F3", INV));
                    }
                    return "ok " + sb4.ToString();
                }

            case "bodyinfo":
                // rb.position vs transform.position: if they disagree the
                // physics is fine and the RENDERING is lying.
                {
                    if (player == null) return "err no_player";
                    var sb3 = new StringBuilder();
                    foreach (var rb in Bodies())
                    {
                        if (rb == null) continue;
                        var tp = rb.transform.position;
                        if (sb3.Length > 0) sb3.Append(';');
                        sb3.Append(rb.name).Append(',')
                           .Append(rb.position.x.ToString("F4", INV)).Append(',')
                           .Append(rb.position.y.ToString("F4", INV)).Append(',')
                           .Append(tp.x.ToString("F4", INV)).Append(',')
                           .Append(tp.y.ToString("F4", INV)).Append(',')
                           .Append(rb.interpolation).Append(',')
                           .Append(tp.z.ToString("F4", INV));
                    }
                    return "ok " + sb3.ToString();
                }

            case "input":
                SetInputEnabled(p.Length > 1 && p[1] == "1");
                return "ok input " + (fiInputEnabled != null && player != null
                                      ? fiInputEnabled.GetValue(player) : "?");

            case "speed":
                Time.timeScale = Flt(p[1]);
                return "ok";

            case "render":
                SetRender(p[1] == "1");
                return "ok";

            case "bench":
                {
                    if (player == null) return "err no_player";
                    if (!lockstep) return "err not_lockstep";
                    int n = p.Length > 1 ? int.Parse(p[1], INV) : 1000;
                    var before = Capture();          // benchmark must not move the pot
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < n; i++)
                    {
                        inManualTick = true;
                        try { miFixedUpdate?.Invoke(player, null); }
                        finally { inManualTick = false; }
                        Physics2D.Simulate(dt);
                    }
                    sw.Stop();
                    Apply(before);
                    return "ok " + (n / sw.Elapsed.TotalSeconds).ToString("F0", INV);
                }

            case "dump":
                {
                    if (player == null) return "err no_player";
                    State snap;
                    if (p.Length > 1)
                    {
                        if (!saves.TryGetValue(p[1], out snap)) return "err no_save";
                    }
                    else snap = Capture();
                    return "ok " + Encode(snap);
                }

            case "restore":
                {
                    if (player == null) return "err no_player";
                    if (p.Length < 3) return "err usage_restore_key_blob";
                    var snap = Decode(p[2]);
                    if (snap == null || snap.bodies.Count == 0) return "err bad_blob";
                    saves[p[1]] = snap;
                    return "ok " + snap.bodies.Count;
                }

            case "teleport":
                {
                    if (player == null) return "err no_player";
                    Teleport(new Vector2(Flt(p[1]), Flt(p[2])));
                    lastArc = -1f;          // re-measure; never trust a supplied arc
                    return "ok " + Obs();
                }

            // Where the ending triggers. Reaching it starts a cutscene that
            // eventually loads another scene and destroys the player, so training
            // needs to stop just short of it.
            case "summit":
                {
                    foreach (var mb in FindObjectsOfType<MonoBehaviour>())
                    {
                        if (mb == null || mb.GetType().Name != "WellDone") continue;
                        var col = mb.GetComponent<Collider2D>();
                        Vector3 c = col != null ? col.bounds.center : mb.transform.position;
                        float arc = -1f;
                        if (spline != null && miNearestTF != null && miTFToDistance != null)
                        {
                            object tf = miNearestTF.Invoke(spline, new object[] { c, 0, -1 });
                            arc = (float)miTFToDistance.Invoke(spline, new object[] { tf });
                        }
                        return string.Format(INV, "ok x={0:F2} y={1:F2} arc={2:F2} name={3}",
                                             c.x, c.y, arc, mb.gameObject.name);
                    }
                    return "err no_welldone";
                }

            case "reloadscene":
                {
                    string name = p.Length > 1 ? p[1]
                        : (gameScene.Length > 0 ? gameScene : SceneManager.GetActiveScene().name);
                    SceneManager.LoadScene(name);
                    return "ok " + name;      // no Obs(): the player is about to be replaced
                }

            // "markers x,y[,kind[,key]];..."  kind 0 usable, 1 dead end,
            // 2 hand-placed, 3 placed by the agent itself when a rung it could
            // not clear got split, 4 the rung this episode started from, 5 the
            // rung it is trying to reach. "markers clear" removes them.
            // Poll for edit requests made in the game window, and clear them.
            // "camfix x y height" locks a wide shot covering `height` world
            // units vertically; "camfix off" gives the camera back to the game.
            case "obspanel":
                // "obspanel 1 [scale]" draws the OBSERVATION / ACTION readout.
                obsPanel = !(p.Length > 1 && p[1] == "0");
                if (p.Length > 2) obsPadScale = Mathf.Max(0.01f, Flt(p[2]));
                return "ok obspanel " + (obsPanel ? "1" : "0")
                       + " scale=" + obsPadScale.ToString("F2", INV);

            case "camfollow":
                // "camfollow <height> [ease]" tracks the pot every rendered
                // frame; "camfollow off" stops.
                if (p.Length > 1 && p[1] == "off")
                {
                    camFollow = false;
                    return "ok camfollow off";
                }
                if (p.Length < 2) return "err usage_camfollow_height";
                if (p.Length > 2) camFollowEase = Mathf.Clamp01(Flt(p[2]));
                camFollow = true;
                camSmooth = root != null ? root.position : Vector2.zero;
                CamFix(true, camSmooth.x, camSmooth.y, Flt(p[1]));
                return string.Format(INV, "ok camfollow h={0:F1} ease={1:F2}",
                                     Flt(p[1]), camFollowEase);

            case "demokey":
                // Which save F restores. "demokey off" disables it.
                demoKey = (p.Length > 1 && p[1] != "off") ? p[1] : null;
                return "ok demokey " + (demoKey ?? "off");

            case "demorec":
                // "demorec 1" starts capturing, "demorec 0" stops,
                // "demorec clear" throws the recording away.
                if (p.Length > 1 && p[1] == "clear")
                {
                    demo.Clear();
                    demoAttempt = 0;
                    return "ok demorec cleared";
                }
                demoRec = !(p.Length > 1 && p[1] == "0");
                return "ok demorec " + (demoRec ? "1" : "0")
                       + " rows=" + demo.Count + " attempts=" + demoAttempt;

            case "demodump":
                // "demodump <from> [n]" -- paged, because a few minutes of
                // 120Hz capture is far more than one line should carry.
                {
                    int at = p.Length > 1 ? int.Parse(p[1], INV) : 0;
                    int want = p.Length > 2 ? int.Parse(p[2], INV) : 200;
                    if (at < 0) at = 0;
                    var sbd = new StringBuilder(4096);
                    int end = Mathf.Min(demo.Count, at + Mathf.Max(1, want));
                    for (int i = at; i < end; i++)
                    {
                        if (sbd.Length > 0) sbd.Append(';');
                        var r = demo[i];
                        for (int k = 0; k < r.Length; k++)
                        {
                            if (k > 0) sbd.Append(',');
                            sbd.Append(r[k].ToString("F4", INV));
                        }
                    }
                    return "ok " + end + "/" + demo.Count + " " + sbd.ToString();
                }

            case "fps":
                // "fps 120" uncaps the render rate; "fps 0" restores.
                //
                // A lockstep shot advances physics only when a step command
                // arrives, and a command is serviced once per rendered frame.
                // So the physics update rate is CAPPED BY THE FRAME RATE, and
                // at 60fps the pot can only move 60 times a second -- half of
                // what a 4-chunk playback asks for, which is why it runs at
                // half speed. Raise the frame rate and it runs at 1x.
                {
                    if (p.Length > 1 && p[1] == "0")
                    {
                        if (fpsSaved)
                        {
                            QualitySettings.vSyncCount = vsyncWas;
                            Application.targetFrameRate = fpsWas;
                            fpsSaved = false;
                        }
                        return string.Format(INV, "ok fps restored now={0:F0}",
                                             fpsAvg);
                    }
                    if (p.Length < 2)
                        return string.Format(INV, "ok fps now={0:F0} target={1} "
                                             + "vsync={2}", fpsAvg,
                                             Application.targetFrameRate,
                                             QualitySettings.vSyncCount);
                    if (!fpsSaved)
                    {
                        vsyncWas = QualitySettings.vSyncCount;
                        fpsWas = Application.targetFrameRate;
                        fpsSaved = true;
                    }
                    QualitySettings.vSyncCount = 0;   // vsync caps at the panel
                    Application.targetFrameRate = int.Parse(p[1], INV);
                    return string.Format(INV, "ok fps target={0} was={1} now={2:F0}",
                                         Application.targetFrameRate, fpsWas,
                                         fpsAvg);
                }

            case "interp":
                // "interp 1" turns rigidbody interpolation back on while in
                // lockstep. Lockstep forces it OFF, because an interpolated
                // transform lags the rigidbody and would corrupt a snapshot --
                // correct for training, but it also means the pot's transform
                // only moves when physics steps. At 30 steps a second against a
                // 60fps render each pose is held for two frames, and a fast
                // swing reads as a smear. For a recording, where nothing is
                // captured mid-run, interpolating is the honest picture of the
                // same motion.
                {
                    bool on = p.Length > 1 && p[1] == "1";
                    int n = 0;
                    foreach (var rb in Bodies())
                    {
                        if (rb == null) continue;
                        rb.interpolation = on
                            ? RigidbodyInterpolation2D.Interpolate
                            : RigidbodyInterpolation2D.None;
                        n++;
                    }
                    return "ok interp " + (on ? "1" : "0") + " bodies=" + n;
                }

            case "blur":
                // "blur 0" makes the pot sharp while it moves; "blur 1" puts
                // the game's own look back.
                //
                // Two culprits, both in Post Processing Stack v2. TEMPORAL
                // anti-aliasing accumulates across frames, and our physics runs
                // in irregular lockstep chunks with hard teleports on reset --
                // so it ghosts badly on every swing. MOTION BLUR smears by
                // design. Both are reached by reflection: this plugin does not
                // reference the post-processing assembly, and should not have
                // to in order to turn two things off.
                {
                    CacheCameras();
                    bool on = p.Length > 1 && p[1] == "1";
                    int aa = 0, mb2 = 0;
                    foreach (var cs in cams)
                    {
                        if (cs == null || cs.cam == null) continue;
                        foreach (var c in cs.cam.GetComponents<MonoBehaviour>())
                        {
                            if (c == null) continue;
                            var tn = c.GetType().Name;
                            if (tn == "PostProcessLayer")
                                aa += SetAntialiasing(c, on) ? 1 : 0;
                            else if (tn == "PostProcessVolume")
                                mb2 += SetMotionBlur(c, on);
                        }
                    }
                    return "ok blur " + (on ? "1" : "0")
                           + " taa_layers=" + aa + " motionblur=" + mb2;
                }

            case "fx":
                // Image effects living on the cameras. "fx list" names them;
                // "fx <substring> 0|1" switches the matching ones off or on.
                // A motion-blur effect smears the pot every time it swings,
                // which is exactly the thing a recording must not do.
                {
                    CacheCameras();
                    if (p.Length < 2 || p[1] == "list")
                    {
                        var sbf = new StringBuilder();
                        foreach (var cs in cams)
                        {
                            if (cs == null || cs.cam == null) continue;
                            foreach (var mb in cs.cam.GetComponents<MonoBehaviour>())
                            {
                                if (mb == null) continue;
                                if (sbf.Length > 0) sbf.Append(';');
                                sbf.Append(mb.GetType().Name)
                                   .Append('=')
                                   .Append(mb.enabled ? '1' : '0');
                            }
                        }
                        return "ok " + (sbf.Length > 0 ? sbf.ToString() : "none");
                    }
                    bool on = p.Length > 2 && p[2] == "1";
                    string want = p[1].ToLowerInvariant();
                    int hit = 0;
                    foreach (var cs in cams)
                    {
                        if (cs == null || cs.cam == null) continue;
                        foreach (var mb in cs.cam.GetComponents<MonoBehaviour>())
                        {
                            if (mb == null) continue;
                            if (!mb.GetType().Name.ToLowerInvariant().Contains(want))
                                continue;
                            mb.enabled = on;
                            hit++;
                        }
                    }
                    return "ok fx " + want + " " + (on ? "1" : "0")
                           + " matched=" + hit;
                }

            case "caminfo":
                // The visible world height the camera is showing RIGHT NOW, in
                // the same units camfix takes. Ask before locking the camera
                // and you get the game's own framing -- which is what a follow
                // shot wants, rather than a height computed to fit a whole
                // ladder on screen.
                {
                    var mc = Camera.main;
                    if (mc == null) return "err no_camera";
                    float hgt = mc.orthographic
                        ? mc.orthographicSize * 2f
                        : 2f * Mathf.Abs(mc.transform.position.z)
                          * Mathf.Tan(mc.fieldOfView * 0.5f * Mathf.Deg2Rad);
                    return string.Format(INV,
                        "ok height={0:F2} ortho={1} locked={2}",
                        hgt, mc.orthographic ? 1 : 0, camLocked ? 1 : 0);
                }

            case "camfix":
                if (p.Length > 1 && p[1] == "off")
                {
                    CamFix(false, 0f, 0f, 0f);
                    return "ok camfix off";
                }
                if (p.Length < 4) return "err usage_camfix_x_y_height";
                CamFix(true, Flt(p[1]), Flt(p[2]), Flt(p[3]));
                return string.Format(INV, "ok camfix {0:F1},{1:F1} h={2:F1}",
                                     Flt(p[1]), Flt(p[2]), Flt(p[3]));

            case "edits":
                {
                    string r = (pendingAdd ? "add " : "") + (pendingDelete ? "del " : "");
                    pendingAdd = pendingDelete = false;
                    r = r.Trim();
                    if (r.Length == 0) r = "none";
                    // the client must not settle a hand-made pose -- that would
                    // undo the very thing being captured
                    return "ok " + r + (manual ? " manual" : "");
                }

            case "flymode":
                flymode = p[1] == "1";
                if (flymode) Logger.LogInfo(
                    "fly mode: arrows move, shift = fast, 8/9 step "
                    + "through " + markers.Count + " checkpoints, space = drop, "
                    + "ENTER = add checkpoint, DELETE = remove nearest, "
                    + "M = play by hand");
                return "ok flymode " + flymode;

            case "markers":
                {
                    markers.Clear();
                    if (p.Length > 1 && p[1] != "clear")
                        foreach (var rec in p[1].Split(';'))
                        {
                            var f = rec.Split(',');
                            if (f.Length < 2) continue;
                            markers.Add(new Marker
                            {
                                p = new Vector2(Flt(f[0]), Flt(f[1])),
                                kind = f.Length > 2 ? int.Parse(f[2], INV) : 0,
                                key = f.Length > 3 ? f[3] : null
                            });
                        }
                    BuildMarkers();
                    return "ok " + markers.Count;
                }

            case "hudtext":
                // "hudtext 0" keeps marker labels but drops the corner readout
                hudText = p.Length > 1 && p[1] == "1";
                return "ok hudtext " + hudText;

            case "isolate":
                SetIsolate(p.Length > 1 && p[1] == "1");
                return "ok isolate " + isolated;

            case "playervis":
                // "playervis 0" hides the real player; 1 brings it back.
                {
                    bool show = p.Length > 1 && p[1] == "1";
                    if (show)
                    {
                        foreach (var r in playerHidden)
                            if (r != null) r.enabled = true;
                        playerHidden.Clear();
                    }
                    else if (playerHidden.Count == 0 && player != null)
                    {
                        foreach (var r in player.transform.root
                                     .GetComponentsInChildren<Renderer>(true))
                        {
                            if (r == null || !r.enabled) continue;
                            if (r is LineRenderer) continue;   // our overlays
                            r.enabled = false;
                            playerHidden.Add(r);
                        }
                    }
                    return "ok playervis " + (playerHidden.Count == 0 ? "1" : "0");
                }

            case "cue":
                // "cue" reports and clears; "cue clear" just clears.
                {
                    bool was = cued;
                    cued = false;
                    return "ok " + (p.Length > 1 && p[1] == "clear"
                                    ? "0" : (was ? "1" : "0"));
                }

            case "poseorder":
                // The transform list every pose refers to, in order. Sent once
                // so per-frame traffic can be pure numbers.
                {
                    CachePoseOrder();
                    var sb6 = new StringBuilder();
                    foreach (var tr in poseOrder)
                    {
                        if (sb6.Length > 0) sb6.Append(';');
                        sb6.Append(tr == null ? "?" : tr.name);
                    }
                    return "ok " + sb6.ToString();
                }

            case "pose":
                // Local position+rotation of every transform, in poseOrder.
                // Local, not world: a bone's local rotation is what actually
                // animates, and it replays onto a clone without needing the
                // parent chain to be identical.
                {
                    CachePoseOrder();
                    var sb7 = new StringBuilder(4096);
                    for (int i = 0; i < poseOrder.Length; i++)
                    {
                        var tr = poseOrder[i];
                        if (tr == null) continue;
                        if (i > 0) sb7.Append(';');
                        var lp = tr.localPosition;
                        var le = tr.localEulerAngles;
                        sb7.Append(lp.x.ToString("F4", INV)).Append(',')
                           .Append(lp.y.ToString("F4", INV)).Append(',')
                           .Append(lp.z.ToString("F4", INV)).Append(',')
                           .Append(le.x.ToString("F2", INV)).Append(',')
                           .Append(le.y.ToString("F2", INV)).Append(',')
                           .Append(le.z.ToString("F2", INV));
                    }
                    return "ok " + sb7.ToString();
                }

            case "ghostrig":
                // "ghostrig N" clones the player N times; "ghostrig 0" clears.
                SetGhostRig(p.Length > 1 ? int.Parse(p[1], INV) : 0);
                return "ok ghostrig " + rigs.Count;

            case "ghostpose":
                // "ghostpose <i> <pose>"  -- pose clone i from a `pose` string
                {
                    if (p.Length < 3) return "err usage_ghostpose_i_pose";
                    int gi = int.Parse(p[1], INV);
                    if (gi < 0 || gi >= rigs.Count) return "err bad_index";
                    ApplyPose(rigs[gi], p[2]);
                    return "ok";
                }

            case "ghostframe":
                // Every clone in one command: "i:pose|i:pose|..." -- one round
                // trip a frame instead of one per ghost, which is the
                // difference between 30fps playback and a slideshow.
                {
                    if (p.Length < 2) return "err usage_ghostframe";
                    foreach (var part in p[1].Split('|'))
                    {
                        int c2 = part.IndexOf(':');
                        if (c2 <= 0) continue;
                        int gi = int.Parse(part.Substring(0, c2), INV);
                        if (gi < 0 || gi >= rigs.Count) continue;
                        ApplyPose(rigs[gi], part.Substring(c2 + 1));
                    }
                    return "ok";
                }

            case "posemask":
                // "posemask 0,3,7,..." -- the transforms a stored frame holds.
                // "posemask clear" goes back to all of them.
                {
                    CachePoseOrder();
                    if (p.Length < 2 || p[1] == "clear")
                    {
                        poseMask = null;
                        return "ok posemask all";
                    }
                    var fm = p[1].Split(',');
                    var mask = new List<int>(fm.Length);
                    foreach (var one in fm)
                    {
                        int v;
                        if (!int.TryParse(one, NumberStyles.Integer, INV, out v))
                            continue;
                        if (v >= 0 && v < poseOrder.Length) mask.Add(v);
                    }
                    poseMask = mask.ToArray();
                    return "ok posemask " + poseMask.Length + " of "
                           + poseOrder.Length;
                }

            case "ghosttrack":
                // "ghosttrack <i> <frames>" reserves clone i's track.
                {
                    if (p.Length < 3) return "err usage_ghosttrack_i_frames";
                    int gi = int.Parse(p[1], INV);
                    int nf = int.Parse(p[2], INV);
                    if (gi < 0 || nf < 0) return "err bad_index";
                    while (ghostTracks.Count <= gi) ghostTracks.Add(null);
                    ghostTracks[gi] = new float[nf][];
                    return "ok ghosttrack " + gi + " " + nf;
                }

            case "ghostload":
                // "ghostload <i> <first> <pose>|<pose>|..." fills frames in,
                // in chunks, so one huge upload does not stall the game loop.
                {
                    if (p.Length < 4) return "err usage_ghostload_i_first_frames";
                    int gi = int.Parse(p[1], INV);
                    int at = int.Parse(p[2], INV);
                    if (gi < 0 || gi >= ghostTracks.Count
                        || ghostTracks[gi] == null) return "err no_track";
                    var trk = ghostTracks[gi];
                    if (at < 0) return "err bad_index";
                    foreach (var frame in p[3].Split('|'))
                    {
                        if (at >= trk.Length) break;
                        trk[at++] = ParseFrame(frame);
                    }
                    return "ok " + at;
                }

            case "ghostdelay":
                // "ghostdelay <i> <frames>" -- clone i stays hidden until then,
                // and its own track starts at that moment rather than at 0.
                {
                    if (p.Length < 3) return "err usage_ghostdelay_i_frames";
                    int gi = int.Parse(p[1], INV);
                    int nf = int.Parse(p[2], INV);
                    if (gi < 0) return "err bad_index";
                    while (ghostDelay.Count <= gi) ghostDelay.Add(0);
                    ghostDelay[gi] = Mathf.Max(0, nf);
                    return "ok ghostdelay " + gi + " " + ghostDelay[gi];
                }

            case "ghostplay":
                // "ghostplay <frame>" -- every model, one short command. A
                // clone whose own track ran out holds its last frame, so a
                // model that gave up early simply stops climbing.
                {
                    if (p.Length < 2) return "err usage_ghostplay_frame";
                    int fr = int.Parse(p[1], INV);
                    int n = Mathf.Min(rigs.Count, ghostTracks.Count);
                    for (int i = 0; i < n; i++)
                    {
                        var trk = ghostTracks[i];
                        if (trk == null || trk.Length == 0) continue;
                        int d = i < ghostDelay.Count ? ghostDelay[i] : 0;
                        bool show = fr >= d;
                        while (ghostShown.Count <= i) ghostShown.Add(true);
                        if (ghostShown[i] != show)
                        {
                            SetRigVisible(rigs[i], show);
                            ghostShown[i] = show;
                        }
                        if (!show) continue;
                        var v = trk[Mathf.Clamp(fr - d, 0, trk.Length - 1)];
                        if (v != null) ApplyFrame(rigs[i], v);
                    }
                    return "ok";
                }

            case "ghosts":
                // "ghosts x,y,kind;x,y,kind;..."  or "ghosts clear"
                {
                    if (p.Length < 2 || p[1] == "clear")
                    {
                        foreach (var lr in ghostPool)
                            if (lr != null) lr.enabled = false;
                        return "ok ghosts 0";
                    }
                    var recs = p[1].Split(';');
                    EnsureGhosts(recs.Length);
                    int used = 0;
                    foreach (var rec in recs)
                    {
                        var f = rec.Split(',');
                        if (f.Length < 2) continue;
                        int kind = f.Length > 2 ? int.Parse(f[2], INV) : 0;
                        DrawGhost(ghostPool[used], Flt(f[0]), Flt(f[1]), kind);
                        used++;
                    }
                    for (int i = used; i < ghostPool.Count; i++)
                        if (ghostPool[i] != null) ghostPool[i].enabled = false;
                    return "ok ghosts " + used;
                }

            case "route":
                // "route <text>" -- shown in the HUD corner. "route" clears it.
                routeLabel = p.Length > 1
                    ? string.Join(" ", p, 1, p.Length - 1) : "";
                return "ok route";

            case "rays":
                {
                    if (player == null) return "err no_player";
                    var sb2 = new StringBuilder(256);
                    sb2.Append("body");
                    CastFan(root.position, RAYS_BODY, RANGE_BODY,
                            v => sb2.Append(' ').Append((v * RANGE_BODY).ToString("F2", INV)));
                    sb2.Append(" tip");
                    CastFan(tipRB != null ? tipRB.position : root.position,
                            RAYS_TIP, RANGE_TIP,
                            v => sb2.Append(' ').Append((v * RANGE_TIP).ToString("F2", INV)));
                    sb2.Append(" mask ").Append(terrainMask.ToString("X8", INV));
                    return "ok " + sb2.ToString();
                }

            // Capture helpers for recording: draw the observation's ray fan in
            // the world, and a minimal on-screen readout.
            case "viz":
                // "viz 0" off, "viz 1"/"viz both", "viz pot", "viz hammer"
                {
                    string m = p.Length > 1 ? p[1] : "0";
                    if (m == "pot" || m == "body") { viz = true; vizMask = 1; }
                    else if (m == "hammer" || m == "tip") { viz = true; vizMask = 2; }
                    else if (m == "both" || m == "1") { viz = true; vizMask = 3; }
                    else viz = false;
                    if (viz) BuildViz();
                    if (vizLines != null && !viz)
                        foreach (var lr in vizLines)
                            if (lr != null) lr.SetPosition(1, lr.GetPosition(0));
                    return "ok viz " + (viz ? (vizMask == 1 ? "pot"
                                             : vizMask == 2 ? "hammer" : "both")
                                            : "off");
                }

            case "hud":
                hud = p[1] == "1";
                return "ok hud " + hud;

            case "window":
                {
                    winBack = Flt(p[1]);
                    winFwd = p.Length > 2 ? Flt(p[2]) : winBack;
                    return "ok";
                }

            case "gravity":
                if (p.Length > 2) Physics2D.gravity = new Vector2(Flt(p[1]), Flt(p[2]));
                return string.Format(INV, "ok {0:R} {1:R}",
                                     Physics2D.gravity.x, Physics2D.gravity.y);

            case "seek":
                lastArc = Flt(p[1]);
                return "ok";

            // Move a real mouse for a few seconds with the agent off, then read
            // this back to see the magnitude range a human actually produces.
            case "calibrate":
                if (p.Length > 1 && p[1] == "reset") humanAxisMax = 0f;
                return "ok humanAxisMax=" + humanAxisMax.ToString("F4", INV)
                       + " sensitivity=" + (fSensitivity != null
                            ? ((float)fSensitivity.GetValue(player)).ToString("F3", INV) : "?");

            case "splineinfo":
                {
                    if (spline == null) return "err no_spline";
                    float len = SplineLength();
                    Vector3 a = SplineAt(0f), b = SplineAt(len);
                    return string.Format(INV,
                        "ok length={0:F2} start={1:F2},{2:F2} end={3:F2},{4:F2}",
                        len, a.x, a.y, b.x, b.y);
                }

            case "splinesample":
                {
                    if (spline == null) return "err no_spline";
                    int n = p.Length > 1 ? int.Parse(p[1], INV) : 64;
                    if (n < 2) return "err need_at_least_2";
                    float len = SplineLength();
                    var sb2 = new StringBuilder(n * 24);
                    for (int i = 0; i < n; i++)
                    {
                        float d = len * i / (n - 1);
                        Vector3 q = SplineAt(d);
                        if (i > 0) sb2.Append(';');
                        sb2.Append(d.ToString("F3", INV)).Append(',');
                        sb2.Append(q.x.ToString("F3", INV)).Append(',');
                        sb2.Append(q.y.ToString("F3", INV));
                    }
                    return "ok " + sb2.ToString();
                }

            case "info":
                return $"ok dt={dt.ToString("R", INV)} lockstep={lockstep} agent={agentControlled} " +
                       $"bodies={(player != null ? Bodies().Count : 0)} cams={cams.Count} " +
                       $"bg={Application.runInBackground} spline={(spline != null ? "yes" : "no")} " +
                       $"win={winBack.ToString("F0", INV)}/{winFwd.ToString("F0", INV)} " +
                       $"sim={Physics2D.simulationMode} " +
                       $"grav={Physics2D.gravity.y.ToString("F1", INV)} " +
                       $"arc={lastArc.ToString("F1", INV)} scene={SceneManager.GetActiveScene().name} " +
                       $"progress={(player != null ? Progress() : 0f).ToString("F2", INV)} " +
                       $"ctrlfields={pcFields.Length} rays={RAYS_BODY}+{RAYS_TIP} " +
                       $"main={(Camera.main != null ? Camera.main.name : "null")}";

            default:
                return "err unknown_command";
        }
    }

    static float Flt(string s) => float.Parse(s, INV);

    // =====================================================================
    // discovery + patching
    // =====================================================================
    void Discover()
    {
        player = null;
        foreach (var mb in FindObjectsOfType<MonoBehaviour>())
            if (mb != null && mb.GetType().GetField("fakeCursorRB", F) != null) { player = mb; break; }
        if (player == null) return;

        var t = player.GetType();
        hj = t.GetField("hj", F)?.GetValue(player) as HingeJoint2D;
        sj = t.GetField("sj", F)?.GetValue(player) as SliderJoint2D;
        cursorRB = t.GetField("fakeCursorRB", F)?.GetValue(player) as Rigidbody2D;
        fMouseInput = t.GetField("mouseInput", F);
        fSensitivity = t.GetField("mouseSensitivity", F);
        root = player.GetComponent<Rigidbody2D>();

        var tipT = t.GetField("tip", F)?.GetValue(player) as Transform;
        tipRB = tipT != null ? tipT.GetComponentInParent<Rigidbody2D>() : null;
        hingeRB = hj != null ? hj.attachedRigidbody : null;

        gameScene = SceneManager.GetActiveScene().name;

        // Everything except the player itself: rays start inside the pot's own
        // colliders and would otherwise report distance zero in every direction.
        int playerLayer = LayerMask.NameToLayer("Player");
        terrainMask = playerLayer >= 0 ? ~(1 << playerLayer) : ~0;
        dt = Time.fixedDeltaTime;
        miFixedUpdate = AccessTools.Method(t, "FixedUpdate");
        CacheCameras();
        FindSpline();

        var stateFields = new List<FieldInfo>();
        foreach (var f in t.GetFields(F))
        {
            var ft = f.FieldType;
            if (NotState.Contains(f.Name)) continue;
            if (ft == typeof(float) || ft == typeof(bool) || ft == typeof(int) ||
                ft == typeof(Vector2) || ft == typeof(Vector3))
                stateFields.Add(f);
        }
        pcFields = stateFields.ToArray();
        fiInputEnabled = t.GetField("input_enabled", F);

        if (!patched && miFixedUpdate != null)
        {
            harmony.Patch(miFixedUpdate,
                prefix: new HarmonyMethod(typeof(Patches).GetMethod(nameof(Patches.Pre),
                    BindingFlags.Static | BindingFlags.Public)));
            patched = true;
            Logger.LogInfo("patched " + t.FullName + ".FixedUpdate");
        }

        if (!axisPatched)
        {
            // FixedUpdate does `mouseInput = new Vector2(GetAxis("mouseX"),
            // GetAxis("mouseY")) * mouseSensitivity` before using it, so writing
            // mouseInput from a prefix is pointless -- it is overwritten a few
            // instructions later. Injecting at GetAxis puts the agent exactly
            // where a human's mouse enters, and every bit of Foddy's smoothing,
            // clamping and PD control downstream stays intact.
            var rewiredPlayer = AccessTools.TypeByName("Rewired.Player");
            var getAxis = rewiredPlayer != null
                ? AccessTools.Method(rewiredPlayer, "GetAxis", new[] { typeof(string) })
                : null;
            if (getAxis != null)
            {
                harmony.Patch(getAxis,
                    prefix: new HarmonyMethod(typeof(Patches).GetMethod(nameof(Patches.AxisPre),
                        BindingFlags.Static | BindingFlags.Public)),
                    postfix: new HarmonyMethod(typeof(Patches).GetMethod(nameof(Patches.AxisPost),
                        BindingFlags.Static | BindingFlags.Public)));
                axisPatched = true;
                Logger.LogInfo("patched Rewired.Player.GetAxis — agent input is live");
            }
            else Logger.LogError("could not find Rewired.Player.GetAxis; agent input will NOT reach the game");
        }
    }

    void FindSpline()
    {
        spline = null; miNearestTF = null; miTFToDistance = null;
        Component pm = null;
        foreach (var mb in FindObjectsOfType<MonoBehaviour>())
            if (mb != null && mb.GetType().Name == "ProgressMeter") { pm = mb; break; }
        if (pm == null) return;

        spline = pm.GetType().GetField("spline", F)?.GetValue(pm);
        if (spline == null) return;

        var st = spline.GetType();
        miNearestTF = st.GetMethod("GetNearestPointTF",
            new[] { typeof(Vector3), typeof(int), typeof(int) });
        miTFToDistance = st.GetMethod("TFToDistance", new[] { typeof(float) });
        miInterpByDist = st.GetMethod("InterpolateByDistance", new[] { typeof(float) });
        miToWorld = st.GetMethod("ToWorldPosition", new[] { typeof(Vector3) });
        miDistToTF = st.GetMethod("DistanceToTF", new[] { typeof(float) });
        piSplineLength = st.GetProperty("Length");
        piSplineCount = st.GetProperty("Count");
    }

    // TFToSegmentIndex is just (int)(tf * Count) clamped -- reproduced here so we
    // do not have to build a CurvyClamping enum value through reflection.
    int SegIndexAt(float arc, int count)
    {
        float tf = (float)miDistToTF.Invoke(spline, new object[] { arc });
        return Mathf.Clamp((int)(tf * count), 0, count - 1);
    }

    // A fan of world-aligned rays. Rays are not rotated with the pot: its
    // rotation is clamped to +/-15 degrees anyway, and a world-aligned fan gives
    // the policy a stable frame to learn in. Returns normalised distance, 1 for
    // "nothing within range".
    void CastFan(Vector2 origin, int count, float range, Action<float> emit)
    {
        for (int i = 0; i < count; i++)
        {
            float a = (2f * Mathf.PI * i) / count;
            var dir = new Vector2(Mathf.Cos(a), Mathf.Sin(a));
            var hit = Physics2D.Raycast(origin, dir, range, terrainMask);
            emit(hit.collider != null ? hit.distance / range : 1f);
        }
    }

    void BuildViz()
    {
        if (vizRoot != null) return;
        var shader = Shader.Find("Sprites/Default") ?? Shader.Find("Unlit/Color");
        if (shader == null) { Logger.LogError("no shader for ray overlay"); return; }
        var mat = new Material(shader);

        vizRoot = new GameObject("goi-bridge-viz");
        DontDestroyOnLoad(vizRoot);
        vizLines = new LineRenderer[RAYS_BODY + RAYS_TIP];
        for (int i = 0; i < vizLines.Length; i++)
        {
            var go = new GameObject("ray" + i);
            go.transform.SetParent(vizRoot.transform);
            var lr = go.AddComponent<LineRenderer>();
            lr.material = mat;
            lr.useWorldSpace = true;
            lr.positionCount = 2;
            lr.numCapVertices = 2;
            lr.sortingOrder = 32000;
            lr.widthMultiplier = i < RAYS_BODY ? 0.10f : 0.07f;
            vizLines[i] = lr;
        }
    }

    void HideFan(ref int slot, int count, Vector2 origin, float depth)
    {
        for (int i = 0; i < count; i++)
        {
            var lr = vizLines[slot++];
            if (lr == null) continue;
            var z = new Vector3(origin.x, origin.y, depth);
            lr.SetPosition(0, z);
            lr.SetPosition(1, z);
        }
    }

    void DrawRayFan(ref int slot, Vector2 origin, int count, float range,
                    Color miss, Color hit, float depth)
    {
        for (int i = 0; i < count; i++)
        {
            float a = (2f * Mathf.PI * i) / count;
            var dir = new Vector2(Mathf.Cos(a), Mathf.Sin(a));
            var rc = Physics2D.Raycast(origin, dir, range, terrainMask);
            float d = rc.collider != null ? rc.distance : range;
            var lr = vizLines[slot++];
            lr.SetPosition(0, new Vector3(origin.x, origin.y, depth));
            lr.SetPosition(1, new Vector3(origin.x + dir.x * d,
                                          origin.y + dir.y * d, depth));
            var c = rc.collider != null ? hit : miss;
            lr.startColor = c;
            lr.endColor = new Color(c.r, c.g, c.b, rc.collider != null ? 0.95f : 0.20f);
        }
    }

    void BuildMarkers()
    {
        if (markerRoot != null) Destroy(markerRoot);
        if (markers.Count == 0) { markerRoot = null; return; }

        var shader = Shader.Find("Sprites/Default") ?? Shader.Find("Unlit/Color");
        if (shader == null) { Logger.LogError("no shader for markers"); return; }
        var mat = new Material(shader);

        markerRoot = new GameObject("goi-bridge-markers");
        DontDestroyOnLoad(markerRoot);
        for (int i = 0; i < markers.Count; i++)
        {
            var m = markers[i];
            var go = new GameObject("m" + i);
            go.transform.SetParent(markerRoot.transform);
            var lr = go.AddComponent<LineRenderer>();
            lr.material = mat;
            lr.useWorldSpace = true;
            lr.positionCount = 5;
            lr.widthMultiplier = 0.28f;
            lr.sortingOrder = 32001;

            // The rung being trained and the one it is aiming at are drawn
            // larger, so they are findable at training zoom without reading
            // every label on screen.
            float r = (m.kind == 4 || m.kind == 5) ? 2.6f : 1.4f;
            lr.widthMultiplier = (m.kind == 4 || m.kind == 5) ? 0.42f : 0.28f;
            lr.SetPosition(0, new Vector3(m.p.x, m.p.y + r, -2f));
            lr.SetPosition(1, new Vector3(m.p.x + r, m.p.y, -2f));
            lr.SetPosition(2, new Vector3(m.p.x, m.p.y - r, -2f));
            lr.SetPosition(3, new Vector3(m.p.x - r, m.p.y, -2f));
            lr.SetPosition(4, new Vector3(m.p.x, m.p.y + r, -2f));

            Color c = m.kind == 1 ? new Color(1f, 0.28f, 0.22f)     // dead end
                    : m.kind == 2 ? new Color(0.35f, 1f, 0.45f)     // hand-placed
                    : m.kind == 3 ? new Color(1f, 0.85f, 0.2f)      // agent-placed
                    : m.kind == 4 ? new Color(1f, 1f, 1f)           // starting HERE
                    : m.kind == 5 ? new Color(1f, 0.45f, 0.9f)      // aiming for THIS
                                  : new Color(0.35f, 0.7f, 1f);     // usable
            lr.startColor = c;
            lr.endColor = c;
        }
        Logger.LogInfo("drew " + markers.Count + " checkpoint markers");
    }

    void CamFix(bool on, float x, float y, float height)
    {
        var cam = Camera.main;
        if (cam == null) return;

        if (camControl == null)
            foreach (var mb in FindObjectsOfType<MonoBehaviour>())
                if (mb != null && mb.GetType().Name == "CameraControl")
                { camControl = mb as Behaviour; break; }

        if (on)
        {
            if (!camLocked)
            {
                camHome = cam.transform.position;
                camHomeOrtho = cam.orthographicSize;
            }
            camLocked = true;
            if (camControl != null) camControl.enabled = false;

            camHeight = Mathf.Max(2f, height);
            ApplyCam(cam, x, y);
        }
        else
        {
            if (camControl != null) camControl.enabled = true;
            if (camLocked)
            {
                cam.transform.position = camHome;
                cam.orthographicSize = camHomeOrtho;
            }
            camLocked = false;
        }
    }

    // Place the locked camera so `camHeight` world units fill the view.
    void ApplyCam(Camera cam, float x, float y)
    {
        if (cam == null) return;
        if (cam.orthographic)
        {
            cam.orthographicSize = Mathf.Max(1f, camHeight * 0.5f);
            camTarget = new Vector3(x, y, camHome.z);
        }
        else
        {
            float d = (camHeight * 0.5f) /
                      Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
            camTarget = new Vector3(x, y, -Mathf.Max(1f, d));
        }
        cam.transform.position = camTarget;
    }

    // Scroll to zoom, arrows or right-drag to pan, C to re-centre on the pot.
    // None of this touches physics, time, or the agent's input -- the trainer
    // drives the world over TCP and never reads the camera, so a run cannot be
    // disturbed by looking at it from somewhere else.
    void CameraControls()
    {
        // flymode owns the arrow keys for moving the pot; do not pan with them
        // at the same time.
        if (!camLocked || flymode) return;
        var cam = Camera.main;
        if (cam == null) return;

        if (Input.GetKeyDown(KeyCode.C) && root != null)
        {
            ApplyCam(cam, root.position.x, root.position.y);
            return;
        }

        float nx = camTarget.x, ny = camTarget.y;
        bool moved = false;

        float scroll = Input.mouseScrollDelta.y;
        // Keyboard zoom as well: while someone is playing, every mouse movement
        // is swinging the hammer, so right-drag and even cursor position are not
        // available for camera work.
        if (Input.GetKey(KeyCode.Equals) || Input.GetKey(KeyCode.KeypadPlus))
            scroll += 3f * Time.unscaledDeltaTime * 20f;
        if (Input.GetKey(KeyCode.Minus) || Input.GetKey(KeyCode.KeypadMinus))
            scroll -= 3f * Time.unscaledDeltaTime * 20f;
        if (Mathf.Abs(scroll) > 0.01f)
        {
            camHeight = Mathf.Clamp(camHeight * Mathf.Pow(0.85f, scroll), 3f, 600f);
            moved = true;
        }

        // Pan speed scales with how far out we are, so it feels the same zoomed
        // onto the pot as it does looking at the whole mountain.
        float rate = camHeight * 0.6f * Time.unscaledDeltaTime;
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
            rate *= 4f;
        if (Input.GetKey(KeyCode.UpArrow)) { ny += rate; moved = true; }
        if (Input.GetKey(KeyCode.DownArrow)) { ny -= rate; moved = true; }
        if (Input.GetKey(KeyCode.LeftArrow)) { nx -= rate; moved = true; }
        if (Input.GetKey(KeyCode.RightArrow)) { nx += rate; moved = true; }

        // Right-drag. The mouse is idle during training anyway: the agent
        // supplies the axes Rewired would otherwise have read from it.
        if (Input.GetMouseButtonDown(1))
        {
            dragging = true;
            dragOrigin = Input.mousePosition;
        }
        if (Input.GetMouseButtonUp(1)) dragging = false;
        if (dragging && Input.GetMouseButton(1))
        {
            Vector3 d = Input.mousePosition - dragOrigin;
            dragOrigin = Input.mousePosition;
            // One screen pixel is this many world units at the current zoom, so
            // the point under the cursor stays under the cursor.
            float perPixel = camHeight / Mathf.Max(1, Screen.height);
            nx -= d.x * perPixel;
            ny -= d.y * perPixel;
            moved = true;
        }

        if (moved) ApplyCam(cam, nx, ny);
    }

    void FixedUpdate()
    {
        // Only while the human has the controls. Under the agent this would
        // record `injected`, which we already know.
        if (!demoRec || agentControlled || player == null || root == null) return;
        if (demo.Count >= DEMO_MAX) return;

        Vector2 rp = root.position, rv = root.velocity;
        Vector2 tp = tipRB != null ? tipRB.position : rp;
        Vector2 tv = tipRB != null ? tipRB.velocity : Vector2.zero;
        Vector2 cp2 = cursorRB != null ? cursorRB.position : rp;
        float pole = (tipRB != null && hingeRB != null)
            ? Mathf.Atan2(tp.y - hingeRB.position.y, tp.x - hingeRB.position.x) : 0f;
        demo.Add(new float[] {
            Time.time, demoAttempt,
            humanAxis.x, humanAxis.y,          // what the hand is doing
            rp.x, rp.y, rv.x, rv.y,
            root.rotation, root.angularVelocity,
            tp.x - rp.x, tp.y - rp.y, tv.x, tv.y,
            cp2.x - rp.x, cp2.y - rp.y,
            sj != null ? sj.jointTranslation : 0f,
        });
    }

    void LateUpdate()
    {
        CameraControls();
        if (camFollow && root != null)
        {
            // Frame-rate independent easing: the same visual smoothing whether
            // the game is running at 60 or 144.
            float a = 1f - Mathf.Pow(1f - camFollowEase,
                                     Mathf.Max(0.0001f, Time.deltaTime) * 60f);
            camSmooth = Vector2.Lerp(camSmooth, root.position, a);
            var fc = Camera.main;
            if (fc != null) ApplyCam(fc, camSmooth.x, camSmooth.y);
        }
        // Re-assert every frame: the game's own camera code runs too, and a
        // single placement gets overwritten.
        if (camLocked)
        {
            var cam = Camera.main;
            if (cam != null) cam.transform.position = camTarget;
        }

        if (!viz || player == null || root == null) return;
        BuildViz();
        if (vizLines == null) return;
        int slot = 0;
        // A hidden fan is collapsed to zero length rather than left stale: a
        // line that stops updating still draws, and reads as a ray that is
        // stuck rather than one that is switched off.
        // Each fan is drawn in ITS OWN body's plane, slightly toward the
        // camera so the lines sit in front of the art rather than inside it.
        float bodyZ = root.transform.position.z - 0.05f;
        var tipT = tipRB != null ? tipRB.transform : root.transform;
        float tipZ = tipT.position.z - 0.05f;
        if ((vizMask & 1) != 0)
            DrawRayFan(ref slot, root.position, RAYS_BODY, RANGE_BODY,
                       new Color(0.35f, 0.62f, 1f), new Color(1f, 0.45f, 0.15f),
                       bodyZ);
        else
            HideFan(ref slot, RAYS_BODY, root.position, bodyZ);
        if ((vizMask & 2) == 0)
        {
            HideFan(ref slot, RAYS_TIP,
                    tipRB != null ? tipRB.position : root.position, tipZ);
            return;
        }
        DrawRayFan(ref slot, tipRB != null ? tipRB.position : root.position,
                   RAYS_TIP, RANGE_TIP,
                   new Color(0.55f, 0.9f, 0.75f), new Color(1f, 0.85f, 0.2f),
                   tipZ);
    }

    // Colour a ghost by how far along it is: cold blue at the back, white at
    // the front, so the leader reads at a glance in a crowded frame.
    static Color GhostColor(int kind)
    {
        if (kind >= 100) return new Color(1f, 1f, 1f);          // the leader
        float t = Mathf.Clamp01(kind / 20f);
        return Color.Lerp(new Color(0.25f, 0.45f, 0.95f),
                          new Color(1f, 0.75f, 0.15f), t);
    }

    void CachePoseOrder()
    {
        if (poseOrder.Length > 0 || player == null) return;
        poseOrder = player.transform.root.GetComponentsInChildren<Transform>(true);
    }

    void SetGhostRig(int n)
    {
        foreach (var r in rigs)
            if (r != null && r.Length > 0 && r[0] != null)
                Destroy(r[0].root.gameObject);
        rigs.Clear();
        ghostTracks.Clear();          // tracks belong to the rig that is going
        ghostDelay.Clear();
        ghostShown.Clear();
        if (rigRoot != null) { Destroy(rigRoot); rigRoot = null; }
        if (n <= 0 || player == null) return;

        CachePoseOrder();
        rigRoot = new GameObject("goi-bridge-rigs");
        DontDestroyOnLoad(rigRoot);
        var src = player.transform.root.gameObject;

        for (int i = 0; i < n; i++)
        {
            var clone = Instantiate(src);
            clone.name = "ghostrig" + i;
            clone.transform.SetParent(rigRoot.transform, true);

            // Anything that simulates, collides, listens or renders the scene
            // has to go: these are puppets, not players. Renderers stay.
            // Strip in dependency order. Unity refuses to remove a
            // Rigidbody2D while any joint still references it, so destroying
            // components in whatever order GetComponentsInChildren returns
            // leaves the bodies and joints alive -- and twelve live ragdolls
            // spawned on top of each other is not a montage, it is an
            // explosion. Joints first, then colliders, then bodies, then the
            // rest; renderers and transforms are all a puppet needs.
            foreach (var j in clone.GetComponentsInChildren<Joint2D>(true))
                if (j != null) DestroyImmediate(j);
            foreach (var col in clone.GetComponentsInChildren<Collider2D>(true))
                if (col != null) DestroyImmediate(col);
            foreach (var rb in clone.GetComponentsInChildren<Rigidbody2D>(true))
                if (rb != null) DestroyImmediate(rb);
            foreach (var c in clone.GetComponentsInChildren<Component>(true))
            {
                if (c == null || c is Transform || c is MeshFilter) continue;
                var tn = c.GetType().Name;
                if (tn == "MeshRenderer" || tn == "SkinnedMeshRenderer") continue;
                try { DestroyImmediate(c); }
                catch (Exception e)
                {
                    Logger.LogWarning("ghostrig could not strip " + tn
                                      + ": " + e.Message);
                }
            }
            // Nothing may remain that simulates. Say so loudly rather than
            // letting a live clone loose in the scene.
            int leftBodies = clone.GetComponentsInChildren<Rigidbody2D>(true).Length;
            int leftJoints = clone.GetComponentsInChildren<Joint2D>(true).Length;
            int leftCols = clone.GetComponentsInChildren<Collider2D>(true).Length;
            if (leftBodies + leftJoints + leftCols > 0)
                Logger.LogError("ghostrig clone still has " + leftBodies
                                + " bodies, " + leftJoints + " joints, "
                                + leftCols + " colliders — it WILL simulate");
            // A clone made while the real player is hidden inherits its
            // disabled renderers and is invisible. Puppets are always visible;
            // hiding the original is a separate decision.
            foreach (var r in clone.GetComponentsInChildren<Renderer>(true))
                if (r != null) r.enabled = true;
            rigs.Add(clone.GetComponentsInChildren<Transform>(true));
        }
        Logger.LogInfo("ghostrig: " + rigs.Count + " clones of "
                       + poseOrder.Length + " transforms");
    }

    // A stored frame is bare numbers: mask-order, six floats a transform.
    static float[] ParseFrame(string frame)
    {
        var recs = frame.Split(';');
        var v = new float[recs.Length * 6];
        for (int i = 0; i < recs.Length; i++)
        {
            var f = recs[i].Split(',');
            if (f.Length < 6) continue;
            int o = i * 6;
            for (int k = 0; k < 6; k++) v[o + k] = Flt(f[k]);
        }
        return v;
    }

    static void SetRigVisible(Transform[] rig, bool on)
    {
        // Walk THIS clone's own transforms. Not rig[0].root -- every clone is
        // parented under one shared "goi-bridge-rigs" object, so .root resolves
        // to that parent and hiding one late arrival hides the entire cast.
        if (rig == null) return;
        foreach (var tr in rig)
        {
            if (tr == null) continue;
            foreach (var r in tr.GetComponents<Renderer>())
                if (r != null && !(r is LineRenderer)) r.enabled = on;
        }
    }

    void ApplyFrame(Transform[] rig, float[] v)
    {
        if (rig == null || v == null) return;
        int n = v.Length / 6;
        for (int k = 0; k < n; k++)
        {
            int idx = poseMask == null
                      ? k : (k < poseMask.Length ? poseMask[k] : -1);
            if (idx < 0 || idx >= rig.Length) continue;
            var tr = rig[idx];
            if (tr == null) continue;
            int o = k * 6;
            tr.localPosition = new Vector3(v[o], v[o + 1], v[o + 2]);
            tr.localEulerAngles = new Vector3(v[o + 3], v[o + 4], v[o + 5]);
        }
    }

    void ApplyPose(Transform[] rig, string pose)
    {
        if (rig == null) return;
        var recs = pose.Split(';');
        int n = Mathf.Min(recs.Length, rig.Length);
        for (int i = 0; i < n; i++)
        {
            var tr = rig[i];
            if (tr == null) continue;
            var f = recs[i].Split(',');
            if (f.Length < 6) continue;
            tr.localPosition = new Vector3(Flt(f[0]), Flt(f[1]), Flt(f[2]));
            tr.localEulerAngles = new Vector3(Flt(f[3]), Flt(f[4]), Flt(f[5]));
        }
    }

    void EnsureGhosts(int n)
    {
        if (ghostRoot == null)
        {
            ghostRoot = new GameObject("goi-bridge-ghosts");
            DontDestroyOnLoad(ghostRoot);
        }
        var shader = Shader.Find("Sprites/Default") ?? Shader.Find("Unlit/Color");
        while (ghostPool.Count < n)
        {
            var go = new GameObject("g" + ghostPool.Count);
            go.transform.SetParent(ghostRoot.transform);
            var lr = go.AddComponent<LineRenderer>();
            lr.material = new Material(shader);
            lr.useWorldSpace = true;
            lr.positionCount = 9;
            lr.widthMultiplier = 0.30f;
            lr.sortingOrder = 32050;
            ghostPool.Add(lr);
        }
    }

    void DrawGhost(LineRenderer lr, float x, float y, int kind)
    {
        if (lr == null) return;
        lr.enabled = true;
        float r = kind >= 100 ? 1.5f : 1.0f;
        lr.widthMultiplier = kind >= 100 ? 0.42f : 0.30f;
        for (int i = 0; i < 9; i++)
        {
            float a = i * Mathf.PI * 2f / 8f;
            lr.SetPosition(i, new Vector3(x + Mathf.Cos(a) * r,
                                          y + Mathf.Sin(a) * r, -2f));
        }
        Color c = GhostColor(kind);
        lr.startColor = c;
        lr.endColor = c;
    }

    // Hide everything that is not the pot or one of our own overlays. Renderers
    // only -- never cameras: disabling a camera nulls Camera.main and breaks the
    // game's autosave irreversibly.
    void SetIsolate(bool on)
    {
        if (on == isolated) return;
        isolated = on;
        var cam = Camera.main;
        if (on)
        {
            hidden.Clear();
            Transform keep = player != null ? player.transform.root : null;
            foreach (var r in FindObjectsOfType<Renderer>())
            {
                if (r == null || !r.enabled) continue;
                if (r is LineRenderer) continue;              // our overlays
                if (keep != null && r.transform.IsChildOf(keep)) continue;
                r.enabled = false;
                hidden.Add(r);
            }
            if (cam != null)
            {
                isoClear = cam.clearFlags;      // put it back exactly as found
                isoBg = cam.backgroundColor;
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = Color.black;
            }
            Logger.LogInfo("isolate: hid " + hidden.Count + " renderers");
        }
        else
        {
            foreach (var r in hidden)
                if (r != null) r.enabled = true;
            hidden.Clear();
            if (cam != null)
            {
                cam.clearFlags = isoClear;
                cam.backgroundColor = isoBg;
            }
            Logger.LogInfo("isolate off");
        }
    }

    static Texture2D Px(Color c)
    {
        var t = new Texture2D(1, 1);
        t.SetPixel(0, 0, c);
        t.Apply();
        t.hideFlags = HideFlags.HideAndDontSave;
        return t;
    }

    void EnsurePanelTextures()
    {
        if (pxFill != null) return;
        pxFill  = Px(new Color(1f, 1f, 1f, 1f));
        pxFrame = Px(new Color(0.72f, 0.79f, 0.86f, 0.45f));   // the track
        pxBack  = Px(new Color(0f, 0f, 0f, 0.30f));
        pxMark  = Px(new Color(0.90f, 0.94f, 0.98f, 0.95f));
        pxHot   = Px(new Color(0.36f, 0.80f, 0.94f, 1f));      // cool accent
    }

    // One labelled bar: a framed track with a tick showing where the value sits
    // between lo and hi. The tick turns amber near either end, which is what
    // makes a wall of bars readable at a glance -- the eye finds the extremes
    // without reading a single number.
    // A thin open track with a tick, and the name to its LEFT. No box, no
    // fill, no background -- the label sitting inside a boxed bar is the one
    // thing that made this read as a copy of the reference.
    void Bar(Rect r, float v, float lo, float hi, float sc)
    {
        float th = Mathf.Max(1f, 2f * sc);
        float mid = r.y + r.height * 0.5f;
        GUI.DrawTexture(new Rect(r.x, mid - th * 0.5f, r.width, th), pxFrame);
        float t = Mathf.Clamp01(Mathf.InverseLerp(lo, hi, v));
        float tw = Mathf.Max(2f, 3f * sc);
        float x = r.x + t * (r.width - tw);
        GUI.DrawTexture(new Rect(x, r.y + 1f, tw, r.height - 2f),
                        Mathf.Abs(t - 0.5f) > 0.42f ? pxHot : pxMark);
    }

    void OnGUI()
    {
        if (obsPanel) DrawObsPanel();
        // Checkpoint ids, drawn over each marker. Without them a diamond cannot
        // be matched to a row in the probe output.
        if (hud && markers.Count > 0)
        {
            var cam = Camera.main;
            if (cam != null)
            {
                var idStyle = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 15,
                    fontStyle = FontStyle.Bold,
                    alignment = TextAnchor.MiddleCenter,
                    normal = { textColor = Color.white }
                };
                for (int i = 0; i < markers.Count; i++)
                {
                    var m = markers[i];
                    var sp = cam.WorldToScreenPoint(new Vector3(m.p.x, m.p.y, 0f));
                    if (sp.z <= 0f) continue;
                    var r = new Rect(sp.x - 40f, Screen.height - sp.y - 34f, 80f, 20f);
                    string label = (m.key != null && m.key.Length > 0) ? m.key : i.ToString();
                    if (m.kind == 4) label = "> " + label + " <";
                    else if (m.kind == 5) label = label + " *";
                    idStyle.normal.textColor =
                        m.kind == 4 ? new Color(1f, 1f, 1f)
                      : m.kind == 5 ? new Color(1f, 0.6f, 0.95f)
                                    : Color.white;
                    var idShadow = new GUIStyle(idStyle)
                    { normal = { textColor = new Color(0f, 0f, 0f, 0.8f) } };
                    GUI.Label(new Rect(r.x + 1f, r.y + 1f, r.width, r.height), label, idShadow);
                    GUI.Label(r, label, idStyle);
                }
            }
        }

        if (!hud || !hudText || player == null) return;
        var st = new GUIStyle(GUI.skin.label)
        {
            fontSize = 22,
            fontStyle = FontStyle.Bold,
            normal = { textColor = Color.white }
        };
        var shadow = new GUIStyle(st) { normal = { textColor = new Color(0, 0, 0, 0.75f) } };
        string text = string.Format(INV,
            "arc {0:F1}\nheight {1:F1}\n{2}{3}{4}",
            lastArc, root.position.y,
            agentControlled ? "AGENT (your mouse is off)" : "human",
            routeLabel.Length > 0 ? "\n" + routeLabel : "",
            camLocked ? "\nscroll = zoom   arrows/right-drag = pan   C = centre"
                      : "");
        GUI.Label(new Rect(26, 24, 400, 140), text, shadow);
        GUI.Label(new Rect(24, 22, 400, 140), text, st);
    }

    // Arc length along the authored spline, in world units. 0 if unavailable.
    float Progress()
    {
        if (spline == null || miNearestTF == null || miTFToDistance == null) return 0f;
        // Rigidbody2D.position is authoritative and updates the instant we write
        // it; transform.position only catches up when the physics step runs, and
        // in lockstep there may not be one between a restore and the next
        // observation. Reading the Transform there projects the pot's PREVIOUS
        // position, which is silent and corrupts every arc downstream.
        Vector3 pos = player.transform.position;
        if (root != null) { pos.x = root.position.x; pos.y = root.position.y; }
        int count = piSplineCount != null ? (int)piSplineCount.GetValue(spline, null) : 0;

        object tf;
        if (winBack < 0f || lastArc < 0f || count <= 0 || miDistToTF == null)
        {
            tf = miNearestTF.Invoke(spline, new object[] { pos, 0, -1 });
        }
        else
        {
            float len = SplineLength();
            int a = SegIndexAt(Mathf.Clamp(lastArc - winBack, 0f, len), count);
            int b = SegIndexAt(Mathf.Clamp(lastArc + winFwd, 0f, len), count);
            if (b < a) b = a;
            tf = miNearestTF.Invoke(spline, new object[] { pos, a, b });
        }

        lastArc = (float)miTFToDistance.Invoke(spline, new object[] { tf });
        return lastArc;
    }

    float SplineLength()
    {
        if (spline == null || piSplineLength == null) return 0f;
        return (float)piSplineLength.GetValue(spline, null);
    }

    // World position on the spline at the given arc length.
    Vector3 SplineAt(float distance)
    {
        if (spline == null || miInterpByDist == null) return Vector3.zero;
        var pos = (Vector3)miInterpByDist.Invoke(spline, new object[] { distance });
        if (miToWorld != null) pos = (Vector3)miToWorld.Invoke(spline, new object[] { pos });
        return pos;
    }

    // Move the whole rig, preserving its pose, and kill all momentum. In
    // lockstep nothing simulates until the next tick, so this is a free-fly:
    // no gravity to fight and no need to make anything kinematic.
    void Teleport(Vector2 target)
    {
        // Compute every destination BEFORE moving anything. Writing a parent's
        // Transform drags its children with it, so reading rb.position inside
        // the loop can return a value the previous iteration already shifted --
        // the delta then gets applied twice, the joint chain is torn apart, and
        // the solver answers with velocities in the thousands.
        var bodies = Bodies();
        Vector2 delta = target - root.position;
        var dest = new Vector2[bodies.Count];
        for (int i = 0; i < bodies.Count; i++) dest[i] = bodies[i].position + delta;

        for (int i = 0; i < bodies.Count; i++)
        {
            var rb = bodies[i];
            rb.position = dest[i];
            rb.velocity = Vector2.zero;
            rb.angularVelocity = 0f;
            SyncTransform(rb, dest[i], rb.rotation);
        }
        Physics2D.SyncTransforms();
    }

    // Keep the Transform in step with a body we just moved, so anything reading
    // transform.position before the next simulation step -- our own projection,
    // and the game's ProgressMeter and Narrator -- sees where the pot actually is.
    static void SyncTransform(Rigidbody2D rb, Vector2 p, float rot, float? z = null)
    {
        var tr = rb.transform;
        var cur = tr.position;
        tr.position = new Vector3(p.x, p.y, z ?? cur.z);
        // Keep the art's X/Y tilt. These are 3D models driven by 2D physics, so
        // forcing Euler(0,0,rot) discards orientation the renderer depends on --
        // and everything parented below then swings out along depth. On a
        // perspective camera that reads as the hammer being half its length,
        // with the simulation itself perfectly correct.
        var e = tr.rotation.eulerAngles;
        tr.rotation = Quaternion.Euler(e.x, e.y, rot);
    }

    public void NoteHumanAxisMax(float v)
    {
        if (v > humanAxisMax) humanAxisMax = v;
    }

    // =====================================================================
    // stepping
    // =====================================================================
    void SetLockstep(bool on)
    {
        lockstep = on;
        EnsureSimulationMode();
        SetInterpolation(on);
    }

    // Unity resets Physics2D.simulationMode out from under us -- a scene load is
    // the usual culprit. If that happens while lockstep is still on, Simulate()
    // is refused with a warning and the world silently stops advancing while the
    // client happily counts steps. Re-assert it before every tick.
    bool simWarned = false;
    void EnsureSimulationMode()
    {
        var want = lockstep ? SimulationMode2D.Script : SimulationMode2D.FixedUpdate;
        if (Physics2D.simulationMode == want) return;
        if (lockstep && !simWarned)
        {
            simWarned = true;
            Logger.LogWarning("Physics2D.simulationMode was " + Physics2D.simulationMode
                              + " while in lockstep — physics was not advancing. Restored.");
        }
        Physics2D.simulationMode = want;
    }

    void Tick(float dx, float dy, int n)
    {
        injected = new Vector2(dx, dy);
        agentControlled = true;
        EnsureSimulationMode();

        for (int i = 0; i < n; i++)
        {
            if (lockstep)
            {
                inManualTick = true;
                try { miFixedUpdate?.Invoke(player, null); }
                finally { inManualTick = false; }
                Physics2D.Simulate(dt);
            }
        }
    }

    // PostProcessLayer.antialiasingMode is an enum: None, FXAA, SMAA, TAA.
    // Dropping to FXAA rather than None keeps edges clean while removing the
    // temporal accumulation that does the smearing.
    bool SetAntialiasing(MonoBehaviour layer, bool restore)
    {
        try
        {
            var f = layer.GetType().GetField("antialiasingMode");
            if (f == null) return false;
            if (restore)
            {
                object was;
                if (aaWas.TryGetValue(layer, out was) && was != null)
                {
                    f.SetValue(layer, was);
                    aaWas.Remove(layer);
                    return true;
                }
                return false;
            }
            if (!aaWas.ContainsKey(layer)) aaWas[layer] = f.GetValue(layer);
            f.SetValue(layer, Enum.ToObject(f.FieldType, 1));   // FXAA
            return true;
        }
        catch (Exception e)
        {
            Logger.LogWarning("antialiasing: " + e.Message);
            return false;
        }
    }

    // A volume's profile holds a list of effect settings, each with an `active`
    // flag. `profile` (not `sharedProfile`) hands back an instance copy, so the
    // asset on disk is never edited.
    int SetMotionBlur(MonoBehaviour vol, bool restore)
    {
        int n = 0;
        try
        {
            var pp = vol.GetType().GetProperty("profile");
            var prof = pp != null ? pp.GetValue(vol, null) : null;
            if (prof == null) return 0;
            var sf = prof.GetType().GetField("settings");
            var list = sf != null ? sf.GetValue(prof) as System.Collections.IList : null;
            if (list == null) return 0;
            foreach (var eff in list)
            {
                if (eff == null) continue;
                if (eff.GetType().Name != "MotionBlur") continue;
                var af = eff.GetType().GetField("active");
                if (af == null) continue;
                if (restore)
                {
                    if (blurOff.Contains(eff)) { af.SetValue(eff, true); blurOff.Remove(eff); n++; }
                }
                else if ((bool)af.GetValue(eff))
                {
                    af.SetValue(eff, false);
                    blurOff.Add(eff);
                    n++;
                }
            }
        }
        catch (Exception e)
        {
            Logger.LogWarning("motion blur: " + e.Message);
        }
        return n;
    }

    void CacheCameras()
    {
        cams.Clear();
        foreach (var c in Resources.FindObjectsOfTypeAll<Camera>())
        {
            if (c == null || !c.gameObject.scene.IsValid()) continue;   // prefab/asset, not live
            cams.Add(new CamState
            {
                cam = c, mask = c.cullingMask,
                clear = c.clearFlags, enabled = c.enabled
            });
        }
    }

    // Rendering is suppressed by culling everything away, never by disabling the
    // camera. Camera.allCameras lists only ENABLED cameras, so disabling them is a
    // one-way door — nothing is left to re-enable. Worse, the game's own autosave
    // (Saviour.Save) dereferences Camera.main unguarded, so a disabled main camera
    // makes it throw a NullReferenceException every single frame.
    void SetRender(bool on)
    {
        if (cams.Count == 0) CacheCameras();
        foreach (var cs in cams)
        {
            if (cs.cam == null) continue;
            if (on) cs.cam.enabled = cs.enabled;      // recover anything left disabled
            cs.cam.cullingMask = on ? cs.mask : 0;
            // SolidColor, not Nothing: an uncleared framebuffer keeps whatever
            // was last drawn and smears, which looks like a broken renderer
            // rather than a deliberately blank one.
            cs.cam.clearFlags = on ? cs.clear : CameraClearFlags.SolidColor;
            if (!on) cs.cam.backgroundColor = Color.black;
        }
        QualitySettings.vSyncCount = on ? 1 : 0;
        Application.targetFrameRate = on ? -1 : 1000;
    }

    // =====================================================================
    // observation
    // =====================================================================
    // The observation, as an array. The panel and the bridge BOTH read this,
    // so what is drawn on screen is by construction the same vector the policy
    // is fed -- a readout computed separately would drift and this shot exists
    // to be believed.
    public const int OBS_BASE = 20;
    public const int OBS_DIM = OBS_BASE + RAYS_BODY + RAYS_TIP;

    float[] ObsArray()
    {
        var o = new float[OBS_DIM];
        if (player == null || root == null) return o;
        Vector2 rp = root.position, rv = root.velocity;
        Vector2 tp = tipRB != null ? tipRB.position : rp;
        Vector2 tv = tipRB != null ? tipRB.velocity : Vector2.zero;
        Vector2 cp = cursorRB != null ? cursorRB.position : rp;
        Vector2 cv = cursorRB != null ? cursorRB.velocity : Vector2.zero;
        float pole = (tipRB != null && hingeRB != null)
            ? Mathf.Atan2(tp.y - hingeRB.position.y, tp.x - hingeRB.position.x) : 0f;
        float slide = sj != null ? sj.jointTranslation : 0f;

        int k = 0;
        o[k++] = rp.x; o[k++] = rp.y; o[k++] = rv.x; o[k++] = rv.y;
        o[k++] = root.rotation; o[k++] = root.angularVelocity;
        o[k++] = tp.x - rp.x; o[k++] = tp.y - rp.y; o[k++] = tv.x; o[k++] = tv.y;
        o[k++] = cp.x - rp.x; o[k++] = cp.y - rp.y; o[k++] = cv.x; o[k++] = cv.y;
        o[k++] = Mathf.Sin(pole); o[k++] = Mathf.Cos(pole); o[k++] = slide;
        o[k++] = hj != null ? hj.motor.motorSpeed : 0f;
        o[k++] = sj != null ? sj.motor.motorSpeed : 0f;
        o[k++] = Progress();
        CastFan(rp, RAYS_BODY, RANGE_BODY, v => o[k++] = v);
        CastFan(tp, RAYS_TIP, RANGE_TIP, v => o[k++] = v);
        return o;
    }

    string Obs()
    {
        if (player == null) return "";
        var oa = ObsArray();
        var osb = new StringBuilder(320);
        for (int i = 0; i < oa.Length; i++)
        {
            osb.Append(oa[i].ToString("R", INV));
            if (i + 1 < oa.Length) osb.Append(' ');
        }
        return osb.ToString();
    }

    // ---------------------------------------------------------------------
    // The readout: what the agent senses, and where it is aiming.
    //
    // Every value comes from ObsArray() -- the same 44 numbers handed to the
    // policy, in the same order. The network sees them after running mean/std
    // normalisation; raw is shown because a normalised number means nothing on
    // screen and the normaliser keeps moving.
    // ---------------------------------------------------------------------

    GUIStyle stTitle, stGroup, stTiny;
    float aimMax = 2f;          // ring radius, grown to whatever the aim reaches
    float styleScale = -1f;

    // The panel is sized from the SCREEN, not in fixed pixels. Laid out at a
    // base size it comes to ~570px, which is a small corner of a 1080p display
    // and unreadable in a recording -- so it is scaled to fill most of the
    // height, whatever the display happens to be.
    float PanelScale()
    {
        const float natural = 570f;
        return Mathf.Clamp(Screen.height * 0.9f / natural, 0.8f, 2.4f);
    }

    void EnsurePanelStyles(float sc)
    {
        if (stTitle != null && Mathf.Abs(styleScale - sc) < 0.01f) return;
        styleScale = sc;
        stTitle = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(21f * sc), fontStyle = FontStyle.Normal,
            normal = { textColor = new Color(0.86f, 0.91f, 0.96f, 0.92f) }
        };
        stGroup = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(12f * sc), fontStyle = FontStyle.Normal,
            normal = { textColor = new Color(0.55f, 0.66f, 0.76f, 0.95f) }
        };
        stTiny = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(10f * sc),
            normal = { textColor = new Color(0.80f, 0.86f, 0.92f, 0.9f) }
        };
    }

    // A ring or a disc, drawn once into a texture. OnGUI has no circle.
    static Texture2D MakeCircle(int n, float r0, float r1, Color c)
    {
        var t = new Texture2D(n, n, TextureFormat.ARGB32, false);
        var px = new Color[n * n];
        float half = n * 0.5f, edge = 1.6f / half;
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float dx = (x + 0.5f - half) / half;
                float dy = (y + 0.5f - half) / half;
                float d = Mathf.Sqrt(dx * dx + dy * dy);
                float a = Mathf.Clamp01((d - r0) / edge)
                        * Mathf.Clamp01((r1 - d) / edge);
                px[y * n + x] = new Color(c.r, c.g, c.b, c.a * a);
            }
        t.SetPixels(px);
        t.Apply();
        t.hideFlags = HideFlags.HideAndDontSave;
        return t;
    }

    void EnsureCircles()
    {
        if (cRing != null) return;
        cRing = MakeCircle(160, 0.92f, 1.00f,
                           new Color(0.78f, 0.85f, 0.92f, 0.55f));
        cDisc = MakeCircle(160, 0.00f, 0.66f,
                           new Color(0.78f, 0.85f, 0.92f, 0.08f));
        cDot  = MakeCircle(48, 0.00f, 1.00f,
                           new Color(0.36f, 0.80f, 0.94f, 1f));
    }

    float Group(float x, float y, float sc, string name)
    {
        GUI.Label(new Rect(x, y + 2f * sc, 300f * sc, 20f * sc), name, stGroup);
        return y + 18f * sc;
    }

    float Row(float x, ref float y, float w, float sc, string label, float v,
              float lo, float hi)
    {
        float lw = 42f * sc;
        GUI.Label(new Rect(x, y - 2f * sc, lw, 16f * sc), label, stTiny);
        Bar(new Rect(x + lw, y, w - lw, 14f * sc), v, lo, hi, sc);
        y += 16f * sc;
        return y;
    }

    // The rays as cells: filled means terrain is close.
    float Cells(float x, float y, float w, float sc, float[] o, int at, int n,
                int per)
    {
        float gap = 2f * sc, ch = 13f * sc, b = Mathf.Max(1f, sc);
        float cw = (w - (per - 1) * gap) / per;
        int rows = Mathf.CeilToInt(n / (float)per);
        for (int i = 0; i < n; i++)
        {
            var r = new Rect(x + (i % per) * (cw + gap),
                             y + (i / per) * (ch + gap), cw, ch);
            GUI.DrawTexture(r, pxFrame);
            GUI.DrawTexture(new Rect(r.x + b, r.y + b, r.width - b * 2f,
                                     r.height - b * 2f), pxBack);
            float f = Mathf.Clamp01(1f - o[at + i]);
            if (f > 0.02f)
                GUI.DrawTexture(
                    new Rect(r.x + b, r.y + b, (r.width - b * 2f) * f,
                             r.height - b * 2f),
                    f > 0.75f ? pxHot : pxMark);
        }
        return y + rows * (ch + gap) + 4f * sc;
    }

    void DrawObsPanel()
    {
        if (player == null || root == null) return;
        // OnGUI runs several times a frame (Layout, then Repaint); only the
        // Repaint pass draws anything.
        if (Event.current == null || Event.current.type != EventType.Repaint)
            return;
        float sc = PanelScale();
        EnsurePanelTextures();
        EnsurePanelStyles(sc);
        EnsureCircles();

        var o = ObsArray();
        float X = 24f * sc, W = 196f * sc;
        float y = 22f * sc;

        GUI.Label(new Rect(X, y, 500f * sc, 44f * sc), "OBSERVATION", stTitle);
        y += 34f * sc;

        y = Group(X, y, sc, "Velocity");
        Row(X, ref y, W, sc, "x", o[2], -20f, 20f);
        Row(X, ref y, W, sc, "y", o[3], -20f, 20f);

        y = Group(X, y, sc, "Rotation");
        Row(X, ref y, W, sc, "rot", Mathf.DeltaAngle(0f, o[4]), -180f, 180f);
        Row(X, ref y, W, sc, "spin", o[5], -500f, 500f);

        y = Group(X, y, sc, "Hammer Head");
        Row(X, ref y, W, sc, "x", o[6], -5f, 5f);
        Row(X, ref y, W, sc, "y", o[7], -5f, 5f);
        Row(X, ref y, W, sc, "vx", o[8], -40f, 40f);
        Row(X, ref y, W, sc, "vy", o[9], -40f, 40f);

        y = Group(X, y, sc, "Aim");
        Row(X, ref y, W, sc, "x", o[10], -8f, 8f);
        Row(X, ref y, W, sc, "y", o[11], -8f, 8f);
        Row(X, ref y, W, sc, "vx", o[12], -40f, 40f);
        Row(X, ref y, W, sc, "vy", o[13], -40f, 40f);

        y = Group(X, y, sc, "Pole");
        Row(X, ref y, W, sc, "sin", o[14], -1f, 1f);
        Row(X, ref y, W, sc, "cos", o[15], -1f, 1f);
        Row(X, ref y, W, sc, "ext", o[16], -3f, 3f);

        y = Group(X, y, sc, "Motors");
        Row(X, ref y, W, sc, "hinge", o[17], -1000f, 1000f);
        Row(X, ref y, W, sc, "slide", o[18], -1000f, 1000f);

        y = Group(X, y, sc, "Position");
        Row(X, ref y, W, sc, "x", o[0], -50f, 20f);
        Row(X, ref y, W, sc, "y", o[1], -10f, 30f);

        y = Group(X, y, sc, "Progress");
        Row(X, ref y, W, sc, "arc", o[19], 0f, 250f);

        y = Group(X, y, sc, "Body Rays");
        y = Cells(X, y, W, sc, o, OBS_BASE, RAYS_BODY, 8);
        y = Group(X, y, sc, "Tip Rays");
        Cells(X, y, W, sc, o, OBS_BASE + RAYS_BODY, RAYS_TIP, 8);

        DrawAimPad(o, sc);
    }

    void DrawAimPad(float[] o, float sc)
    {
        // The action shown as AIM: where the invisible cursor sits relative to
        // the pot. This is the right thing to draw because it is a POSITION,
        // not a per-frame delta -- it reads identically whether a human or a
        // policy is driving, and needs no calibration between them. The raw
        // mouse axis does, which is why showing that never worked.
        float ax = o[10], ay = o[11];
        float mag = Mathf.Sqrt(ax * ax + ay * ay);
        if (mag > aimMax) aimMax = mag;          // the ring is the reach
        float R = Mathf.Max(0.5f, aimMax);

        var title = new GUIStyle(stTitle) { alignment = TextAnchor.MiddleCenter };
        // Middle right: level with the pot, opposite the observation column.
        float rad = 78f * sc;
        float cx = Screen.width - rad - 90f * sc;
        float cy = Screen.height * 0.5f;
        GUI.Label(new Rect(cx - 200f * sc, cy - rad - 62f * sc, 400f * sc,
                           44f * sc), "ACTION", title);

        GUI.DrawTexture(new Rect(cx - rad, cy - rad, rad * 2f, rad * 2f), cDisc);
        GUI.DrawTexture(new Rect(cx - rad, cy - rad, rad * 2f, rad * 2f), cRing);
        GUI.DrawTexture(new Rect(cx - rad * 0.34f, cy - rad * 0.34f,
                                 rad * 0.68f, rad * 0.68f), cRing);

        float dotr = 9f * sc;
        float px = cx + (ax / R) * (rad - dotr);
        float py = cy - (ay / R) * (rad - dotr);
        GUI.DrawTexture(new Rect(px - dotr, py - dotr, dotr * 2f, dotr * 2f), cDot);

    }

    // =====================================================================
    // save states
    // =====================================================================
    // Interpolation has to be off while we drive physics by hand, and exactly
    // as it was when we hand the game back.
    void SetInterpolation(bool manual)
    {
        if (player == null) return;
        if (manual)
        {
            foreach (var rb in Bodies())
            {
                if (rb == null || interpWas.ContainsKey(rb)) continue;
                interpWas[rb] = rb.interpolation;
                rb.interpolation = RigidbodyInterpolation2D.None;
            }
        }
        else
        {
            foreach (var kv in interpWas)
                if (kv.Key != null) kv.Key.interpolation = kv.Value;
            interpWas.Clear();
        }
    }

    void SetInputEnabled(bool on)
    {
        if (fiInputEnabled == null || player == null) return;
        try { fiInputEnabled.SetValue(player, on); }
        catch (Exception e) { Logger.LogWarning("input_enabled: " + e.Message); }
    }

    List<Rigidbody2D> Bodies()
    {
        var list = new List<Rigidbody2D>(player.transform.root.GetComponentsInChildren<Rigidbody2D>(true));
        if (cursorRB != null && !list.Contains(cursorRB)) list.Add(cursorRB);
        return list;
    }

    State Capture()
    {
        var st = new State();
        foreach (var rb in Bodies())
            st.bodies.Add(new Snap
            {
                rb = rb, name = rb.name,
                p = rb.position, v = rb.velocity,
                r = rb.rotation, w = rb.angularVelocity,
                z = rb.transform.position.z
            });

        st.ctrl = new object[pcFields.Length];
        for (int i = 0; i < pcFields.Length; i++)
            st.ctrl[i] = pcFields[i].GetValue(player);

        st.gravity = Physics2D.gravity;
        st.hasGravity = true;
        return st;
    }

    void Apply(State st)
    {
        foreach (var s in st.bodies)
        {
            if (s.rb == null) continue;
            s.rb.position = s.p;
            s.rb.rotation = s.r;
            s.rb.velocity = s.v;
            s.rb.angularVelocity = s.w;
            SyncTransform(s.rb, s.p, s.r, s.z);
        }
        if (st.ctrl != null)
            for (int i = 0; i < pcFields.Length && i < st.ctrl.Length; i++)
                if (st.ctrl[i] != null) pcFields[i].SetValue(player, st.ctrl[i]);
        if (st.hasGravity) Physics2D.gravity = st.gravity;
        Physics2D.SyncTransforms();
    }

    void Save(string key) { saves[key] = Capture(); }

    bool Load(string key)
    {
        if (!saves.TryGetValue(key, out var st)) return false;
        Apply(st);
        return true;
    }

    // Text form of a snapshot, so Python can keep checkpoints on disk and push
    // them back after the game is restarted. Bodies are keyed by name; the blob
    // contains no spaces because the command protocol is space-delimited.
    string Encode(State st)
    {
        var snap = st.bodies;
        var sb = new StringBuilder(768);
        for (int i = 0; i < snap.Count; i++)
        {
            var s = snap[i];
            if (i > 0) sb.Append(';');
            sb.Append(s.name).Append(',');
            sb.Append(s.p.x.ToString("R", INV)).Append(',');
            sb.Append(s.p.y.ToString("R", INV)).Append(',');
            sb.Append(s.r.ToString("R", INV)).Append(',');
            sb.Append(s.v.x.ToString("R", INV)).Append(',');
            sb.Append(s.v.y.ToString("R", INV)).Append(',');
            sb.Append(s.w.ToString("R", INV));
            if (s.z.HasValue)
                sb.Append(',').Append(s.z.Value.ToString("R", INV));
        }

        // controller state after a '|', so old body-only blobs still decode
        sb.Append('|');
        if (st.ctrl != null)
            for (int i = 0; i < pcFields.Length && i < st.ctrl.Length; i++)
            {
                if (st.ctrl[i] == null) continue;
                if (i > 0) sb.Append(';');
                sb.Append(pcFields[i].Name).Append('=').Append(EncVal(st.ctrl[i]));
            }

        sb.Append('|');
        if (st.hasGravity)
            sb.Append(st.gravity.x.ToString("R", INV)).Append(',')
              .Append(st.gravity.y.ToString("R", INV));
        return sb.ToString();
    }

    static string EncVal(object v)
    {
        if (v is float f) return f.ToString("R", INV);
        if (v is bool b) return b ? "1" : "0";
        if (v is int n) return n.ToString(INV);
        if (v is Vector2 v2) return v2.x.ToString("R", INV) + ":" + v2.y.ToString("R", INV);
        if (v is Vector3 v3) return v3.x.ToString("R", INV) + ":" + v3.y.ToString("R", INV)
                                  + ":" + v3.z.ToString("R", INV);
        return "";
    }

    static object DecVal(Type t, string s)
    {
        if (t == typeof(float)) return Flt(s);
        if (t == typeof(bool)) return s == "1";
        if (t == typeof(int)) return int.Parse(s, INV);
        var p = s.Split(':');
        if (t == typeof(Vector2) && p.Length == 2) return new Vector2(Flt(p[0]), Flt(p[1]));
        if (t == typeof(Vector3) && p.Length == 3) return new Vector3(Flt(p[0]), Flt(p[1]), Flt(p[2]));
        return null;
    }

    State Decode(string blob)
    {
        var parts = blob.Split('|');

        var byName = new Dictionary<string, Rigidbody2D>();
        foreach (var rb in Bodies())
            if (!byName.ContainsKey(rb.name)) byName[rb.name] = rb;

        var st = new State();
        foreach (var rec in parts[0].Split(';'))
        {
            var f = rec.Split(',');
            if (f.Length < 7) return null;
            if (!byName.TryGetValue(f[0], out var rb)) continue;   // body no longer exists
            st.bodies.Add(new Snap
            {
                rb = rb, name = f[0],
                p = new Vector2(Flt(f[1]), Flt(f[2])), r = Flt(f[3]),
                v = new Vector2(Flt(f[4]), Flt(f[5])), w = Flt(f[6]),
                z = f.Length > 7 ? (float?)Flt(f[7]) : null
            });
        }

        if (parts.Length > 1 && parts[1].Length > 0)
        {
            var byField = new Dictionary<string, int>();
            for (int i = 0; i < pcFields.Length; i++) byField[pcFields[i].Name] = i;

            st.ctrl = new object[pcFields.Length];
            foreach (var rec in parts[1].Split(';'))
            {
                int eq = rec.IndexOf('=');
                if (eq <= 0) continue;
                string nm = rec.Substring(0, eq);
                if (!byField.TryGetValue(nm, out int idx)) continue;
                st.ctrl[idx] = DecVal(pcFields[idx].FieldType, rec.Substring(eq + 1));
            }
        }

        if (parts.Length > 2 && parts[2].Length > 0)
        {
            var g = parts[2].Split(',');
            if (g.Length == 2) { st.gravity = new Vector2(Flt(g[0]), Flt(g[1])); st.hasGravity = true; }
        }
        return st;
    }

    // =====================================================================
    // socket server (background thread)
    // =====================================================================
    void NetLoop()
    {
        Thread.CurrentThread.CurrentCulture = INV;
        TcpListener listener = null;
        try
        {
            listener = new TcpListener(IPAddress.Loopback, PORT);
            listener.Start();
        }
        catch (Exception e)
        {
            Debug.LogError("[bridge] listen failed: " + e.Message);
            return;
        }

        while (running)
        {
            TcpClient client = null;
            try
            {
                if (!listener.Pending()) { Thread.Sleep(5); continue; }
                client = listener.AcceptTcpClient();
                client.NoDelay = true;
                var stream = client.GetStream();
                var reader = new StreamReader(stream, Encoding.ASCII);
                var writer = new StreamWriter(stream, Encoding.ASCII) { AutoFlush = true, NewLine = "\n" };

                string line;
                while (running && (line = reader.ReadLine()) != null)
                {
                    if (line.Length == 0) continue;
                    pendingCmd = line;
                    reqReady.Set();
                    if (!resReady.WaitOne(20000)) { writer.WriteLine("err timeout"); continue; }
                    writer.WriteLine(pendingResult);
                }
            }
            catch (Exception e)
            {
                Debug.Log("[bridge] client ended: " + e.Message);
            }
            finally
            {
                try { client?.Close(); } catch { }
            }
        }
        try { listener.Stop(); } catch { }
    }
}

public static class Patches
{
    // Rewired is where the mouse enters the controller. Returning false replaces
    // the real axis with the agent's action for the two mouse axes only;
    // everything else passes through untouched.
    public static bool AxisPre(string actionName, ref float __result)
    {
        var b = Bridge.I;
        if (b == null || !b.agentControlled) return true;
        if (actionName == "mouseX") { __result = b.Injected.x; return false; }
        if (actionName == "mouseY") { __result = b.Injected.y; return false; }
        return true;
    }

    public static void AxisPost(string actionName, float __result)
    {
        var b = Bridge.I;
        if (b == null || b.agentControlled) return;
        if (actionName == "mouseX" || actionName == "mouseY")
        {
            b.NoteHumanAxisMax(Mathf.Abs(__result));
            b.NoteHumanAxis(actionName, __result);
        }
    }

    // Prefix on PlayerControl.FixedUpdate.
    // In lockstep mode Unity's own FixedUpdate calls are suppressed so the
    // controller runs exactly once per manual tick. Returns false to skip.
    public static bool Pre()
    {
        var b = Bridge.I;
        if (b == null || b.player == null) return true;
        if (b.lockstep && !Bridge.inManualTick) return false;
        return true;
    }
}