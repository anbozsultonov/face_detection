import os
import sys
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Получаем абсолютный путь к папке, где лежит migrate.py
project_root = os.path.dirname(os.path.abspath(__file__))
# Добавляем этот путь в список путей поиска модулей
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib.util
from pathlib import Path

def run_migrations():
    migrations_dir = Path("migrations")
    done_file = Path(".migrations.done")

    if not migrations_dir.exists():
        print(f"Error: Folder '{migrations_dir}' not found.")
        return

    if not done_file.exists():
        done_file.touch()

    applied_files = set(done_file.read_text(encoding='utf-8').splitlines())

    # Ищем все файлы .py, которых нет в списке .done
    new_migrations = sorted([
        f for f in migrations_dir.glob("*.py")
        if f.name not in applied_files and f.name != "__init__.py"
    ])

    if not new_migrations:
        print("No new migrations found.")
        return

    for mig_file in new_migrations:
        print(f"🚀 Running migration: {mig_file.name}")

        try:
            # Динамическая загрузка файла миграции
            spec = importlib.util.spec_from_file_location(mig_file.stem, mig_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Вызываем run() без передачи аргументов
            if hasattr(module, 'run'):
                module.run()

                # Записываем в лог только после успешного выполнения функции
                with open(done_file, 'a', encoding='utf-8') as f:
                    f.write(f"{mig_file.name}\n")
                print(f"✅ Finished: {mig_file.name}")
            else:
                print(f"   ⚠️ Warning: No run() function in {mig_file.name}. Skipping.")

        except Exception as e:
            print(f"\n❌ Error in {mig_file.name}:")
            print(f"Details: {e}")
            sys.exit(1)


if __name__ == "__main__":
    run_migrations()