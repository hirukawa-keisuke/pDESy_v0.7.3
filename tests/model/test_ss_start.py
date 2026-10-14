"""SS constrains actual first starts, independently of resource availability."""
import pytest

from pDESy.model.base_project import BaseProject, BaseProjectStatus
from pDESy.model.base_task import BaseTaskState, BaseTaskDependency
from pDESy.model.base_priority_rule import TaskPriorityRuleMode


def model(*, immediate=False, group=False, auto=False):
    p = BaseProject()
    wf = p.create_workflow('workflow')
    a = wf.create_task('a', default_work_amount=3)
    b = wf.create_task('b', default_work_amount=1, auto_task=auto,
                       must_start_immediately=immediate)
    b.add_input_task(a, BaseTaskDependency.SS)
    if group:
        a.parallel_group_id = b.parallel_group_id = 'group'
    team = p.create_team('team')
    team.update_targeted_task_set({a, b})
    wa = team.create_worker('wa', workamount_skill_mean_map={'a': 1})
    wb = team.create_worker('wb', workamount_skill_mean_map={'b': 1})
    return p, a, b, wa, wb


def first_start(task):
    return next(i for i, state in enumerate(task.state_record_list)
                if state == BaseTaskState.WORKING)


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('immediate', [False, True])
def test_successor_waits_for_predecessor_resources(auto, immediate):
    p, a, b, wa, _ = model(auto=auto, immediate=immediate)
    wa.absence_time_list = [0, 1]
    p.simulate(max_time=10, task_priority_rule=TaskPriorityRuleMode.SPT)
    assert p.status == BaseProjectStatus.FINISHED_SUCCESS
    assert first_start(a) == first_start(b) == 2
    assert all(not pairs for pairs in
               b.allocated_worker_facility_id_tuple_set_record_list[:2])


def test_predecessor_does_not_wait_for_successor_resources():
    p, a, b, _, wb = model()
    wb.absence_time_list = [0, 1]
    p.simulate(max_time=10)
    assert first_start(a) == 0
    assert first_start(b) == 2


def test_immediate_failure_occurs_when_predecessor_can_start():
    p, a, b, wa, wb = model(immediate=True)
    wa.absence_time_list = [0, 1]
    wb.absence_time_list = [0, 1, 2]
    with pytest.warns(UserWarning, match='could not be allocated at step 2'):
        p.simulate(max_time=10)
    assert p.time == 2
    assert p.status == BaseProjectStatus.FINISHED_FAILURE


@pytest.mark.parametrize('auto', [False, True])
def test_parallel_group_with_internal_ss_starts_atomically(auto):
    p, a, b, wa, _ = model(group=True, auto=auto)
    wa.absence_time_list = [0, 1]
    p.simulate(max_time=10)
    assert first_start(a) == first_start(b) == 2


def test_ss_chain_across_workflows_starts_in_same_step():
    p, a, b, _, _ = model()
    wf = p.create_workflow('other')
    c = wf.create_task('c', auto_task=True, default_work_amount=1)
    c.add_input_task(b, BaseTaskDependency.SS)
    p.simulate(max_time=10, task_priority_rule=TaskPriorityRuleMode.SPT)
    assert first_start(a) == first_start(b) == first_start(c) == 0


def test_started_predecessor_can_be_absent_when_successor_starts():
    p, a, b, wa, wb = model()
    wa.absence_time_list = [1, 2, 3]
    wb.absence_time_list = [0, 1]
    p.simulate(max_time=10)
    assert first_start(a) == 0
    assert first_start(b) == 2
    assert a.state_record_list[2] == BaseTaskState.READY


def test_external_ss_blocks_whole_parallel_group():
    p, a, b, wa, _ = model()
    c = next(iter(p.workflow_set)).create_task('c', auto_task=True, default_work_amount=1)
    b.parallel_group_id = c.parallel_group_id = 'group'
    wa.absence_time_list = [0, 1]
    p.simulate(max_time=10)
    assert first_start(a) == first_start(b) == first_start(c) == 2


def test_continuation_preserves_predecessor_first_start():
    p, a, b, wa, wb = model()
    wa.absence_time_list = [1, 2, 3]
    wb.absence_time_list = [0, 1]
    with pytest.warns(UserWarning, match='Time Over'):
        p.simulate(max_time=2)
    p.simulate(max_time=10, initialize_state_info=False, initialize_log_info=False)
    assert p.status == BaseProjectStatus.FINISHED_SUCCESS
    assert first_start(a) == 0
    assert first_start(b) == 2


def test_unstarted_predecessor_blocks_auto_successor_until_timeout():
    p, a, b, wa, _ = model(auto=True, immediate=True)
    wa.workamount_skill_mean_map = {}
    with pytest.warns(UserWarning, match='Time Over'):
        p.simulate(max_time=3)
    assert all(state != BaseTaskState.WORKING for state in b.state_record_list)
    assert p.time == 3


@pytest.mark.parametrize('manual', [False, True])
def test_parallel_auto_predecessor_waits_for_entire_group(manual):
    p = BaseProject()
    wf = p.create_workflow('workflow')
    timer = wf.create_task('timer', auto_task=True, default_work_amount=2,
                           parallel_group_id='launch')
    task = wf.create_task('task', auto_task=not manual, default_work_amount=1,
                          parallel_group_id='launch')
    task.add_input_task(timer, BaseTaskDependency.SS)
    if manual:
        team = p.create_team('team')
        team.update_targeted_task_set({task})
        team.create_worker('worker', workamount_skill_mean_map={'task': 1},
                           absence_time_list=[0, 1])
    p.simulate(max_time=10)
    assert p.status == BaseProjectStatus.FINISHED_SUCCESS
    assert first_start(timer) == first_start(task) == (2 if manual else 0)
    assert not timer.allocated_worker_facility_id_tuple_set
