import { describe, expect, it, vi } from "vitest";
import { api } from "./api";

describe("demo API client", () => {
  it("loads scenarios through one typed client", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => [{ id: "x" }] }));
    await expect(api.scenarios()).resolves.toEqual([{ id: "x" }]);
  });
  it("surfaces API errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    await expect(api.scenarios()).rejects.toThrow("503");
  });
});
