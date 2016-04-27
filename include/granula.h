#ifndef _GRANULA_H_
#define _GRANULA_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <time.h>

static long int op_cnt = 0;

struct granula_op {
  long int operationUuid;
  const char *actor_type;
  const char *actor_id;
  const char *mission_type;
  const char *mission_id;
};
typedef struct granula_op granula_op_t;

static long int granula_get_uuid() {
  return __sync_fetch_and_add(&op_cnt, 1);
}

static void granula_get_opinfo(
            char *buf, granula_op_t *gop,
            const char *infoName, const char *infoValue) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  long unsigned timestamp = ts.tv_sec * 1000 + ts.tv_nsec / 1000000; // ms
  sprintf(buf, "GRANULA - OperationUuid: %lu "
               "ActorType: %s "
               "ActorId: %s "
               "MissionType: %s "
               "MissionId: %s "
               "InfoName: %s "
               "InfoValue: %s "
               "Timestamp: %lu",
          granula_get_uuid(),
          gop->actor_type,
          gop->actor_id,
          gop->mission_type,
          gop->mission_id,
          infoName,
          infoValue,
          timestamp);
}

#ifdef __cplusplus
}
#endif

#endif

