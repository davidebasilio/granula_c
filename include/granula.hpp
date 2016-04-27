#ifndef _GRANULA_HPP_
#define _GRANULA_HPP_

#include <chrono>
#include <string>
#include <cstdlib>

namespace granula {
  class operation {
    public:
      std::string operationUuid;
      std::string actor_type;
      std::string actor_id;
      std::string mission_type;
      std::string mission_id;

      operation(std::string a_type, std::string a_id, std::string m_type, std::string m_id) {
        operationUuid = generateUuid();
        actor_type = a_type;
        actor_id = a_id;
        mission_type = m_type;
        mission_id = m_id;
      }

      std::string generateUuid() {
        long uuid;
        if (sizeof(int) < sizeof(long))
          uuid = (static_cast<long>(std::rand()) << (sizeof(int) * 8)) | rand();
        return std::to_string(uuid);
      }

      std::string getOperationInfo(std::string infoName, std::string infoValue) {
        return "GRANULA - OperationUuid:" + operationUuid + " " +
          "ActorType:" + actor_type + " " +
          "ActorId:" + actor_id + " " +
          "MissionType:" + mission_type + " " +
          "MissionId:" + mission_id + " " +
          "InfoName:" + infoName + " " +
          "InfoValue:" + infoValue + " " +
          "Timestamp:" + getEpoch();
      }

      std::string getEpoch() {
        return std::to_string(
            std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count());
      }
  };
}

#endif

