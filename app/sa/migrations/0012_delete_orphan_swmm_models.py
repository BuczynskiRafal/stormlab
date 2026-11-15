"""Delete the uploaded-model rows no session refers to.

Deleting sessions used to leave their ``SWMMModel`` behind; the views now delete it with
its last session (``SessionDeletionService``). This removes the ones left so far. Only the
row goes: the file itself was never stored.
"""

from django.db import migrations


def delete_orphan_swmm_models(apps, schema_editor):
    SWMMModel = apps.get_model("sa", "SWMMModel")
    SWMMModel.objects.filter(calculationsession__isnull=True).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("sa", "0011_session_status_updated_index"),
    ]

    operations = [
        migrations.RunPython(delete_orphan_swmm_models, migrations.RunPython.noop),
    ]
