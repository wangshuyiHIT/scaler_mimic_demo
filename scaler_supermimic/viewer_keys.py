"""Held height keys for MuJoCo's native GLFW viewer, without OS key repeat."""
from threading import Lock


class HeldHeightKeys:
    UP, DOWN = 265, 264

    def __init__(self):
        self._pressed = set()
        self._lock = Lock()

    def event(self, key, action):
        if key not in (self.UP, self.DOWN):
            return
        with self._lock:
            if action == 1:  # GLFW_PRESS; repeat events do not add increments.
                self._pressed.add(key)
            elif action == 0:  # GLFW_RELEASE
                self._pressed.discard(key)

    def clear(self):
        with self._lock:
            self._pressed.clear()

    def drive(self, morph, dt):
        with self._lock:
            direction = int(self.UP in self._pressed) - int(self.DOWN in self._pressed)
        if direction:
            # Advance only this tick: release leaves no queued height travel.
            morph.target = max(-1., min(1., morph.fraction + direction * .8 * dt))


class NativeKeyBridge:
    """Preserve MuJoCo callbacks and additionally observe release/focus events.

    MuJoCo invokes its Python key callback on the GLFW render thread for PRESS
    only. The first such callback gives us the current window/context. Install
    callbacks there, retaining and forwarding the original C++ function pointers.
    GLFW's Python set_key_callback wrapper cannot return a foreign C++ callback,
    so use its typed native entry points (same library used by mujoco.viewer).
    Keep this object alive until the native window has been destroyed/joined.
    """
    def __init__(self, held, on_press):
        self.held, self.on_press = held, on_press
        self._key_callback = self._focus_callback = None
        self._previous_key = self._previous_focus = None

    def __call__(self, key):
        if self._key_callback is None:
            self._install()
        self.held.event(key, 1)
        if key not in (self.held.UP, self.held.DOWN):
            self.on_press(key)

    def _install(self):
        import glfw

        window = glfw.get_current_context()
        if not window:
            raise RuntimeError("Native height keys require the viewer's GLFW render context")

        def key_event(window, key, scancode, action, mods):
            self.held.event(key, action)
            if self._previous_key:
                self._previous_key(window, key, scancode, action, mods)

        def focus_event(window, focused):
            if not focused:
                self.held.clear()
            if self._previous_focus:
                self._previous_focus(window, focused)

        self._key_callback = glfw._GLFWkeyfun(key_event)
        self._focus_callback = glfw._GLFWwindowfocusfun(focus_event)
        self._previous_key = glfw._glfw.glfwSetKeyCallback(window, self._key_callback)
        self._previous_focus = glfw._glfw.glfwSetWindowFocusCallback(window, self._focus_callback)
