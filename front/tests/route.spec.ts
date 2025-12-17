import { test, expect } from "@playwright/test";

test.describe("Маршрут и карта", () => {
  test("строит маршрут и показывает ETA/дистанцию", async ({ page }) => {
    await page.goto("/");

    await page.getByLabel("Откуда").fill("43.238949,76.889709");
    await page.getByLabel("Куда").fill("43.256700,76.928600");
    await page.getByLabel("Время выезда").fill("2025-12-10T15:00");

    await page.getByRole("button", { name: "Построить маршрут" }).click();

    await expect(page.getByText("В пути:")).toBeVisible();
    await expect(page.getByText("Дистанция")).toBeVisible();
    await expect(page.getByText("Длительность")).toBeVisible();
  });

  test("показывает ошибку при 500", async ({ page }) => {
    await page.route("**/api/simulate", (route) =>
      route.fulfill({
        status: 500,
        body: "Internal error"
      })
    );

    await page.goto("/");
    await page.getByLabel("Откуда").fill("43.238949,76.889709");
    await page.getByLabel("Куда").fill("43.256700,76.928600");
    await page.getByLabel("Время выезда").fill("2025-12-10T15:00");

    await page.getByRole("button", { name: "Построить маршрут" }).click();

    await expect(page.getByText("Не удалось получить маршрут")).toBeVisible();
  });
});










