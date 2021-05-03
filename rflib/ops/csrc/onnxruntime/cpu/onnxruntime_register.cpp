#include "onnxruntime_register.h"

#include "corner_pool.h"
#include "grid_sample.h"
#include "nms.h"
#include "ort_rflib_utils.h"
#include "roi_align.h"
#include "roi_align_rotated.h"
#include "soft_nms.h"

const char *c_RFLIBOpDomain = "rflib";
SoftNmsOp c_SoftNmsOp;
NmsOp c_NmsOp;
RFLIBRoiAlignCustomOp c_RFLIBRoiAlignCustomOp;
RFLIBRoIAlignRotatedCustomOp c_RFLIBRoIAlignRotatedCustomOp;
GridSampleOp c_GridSampleOp;
RFLIBCornerPoolCustomOp c_RFLIBCornerPoolCustomOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_RFLIBOpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_NmsOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_RFLIBRoiAlignCustomOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_RFLIBRoIAlignRotatedCustomOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_GridSampleOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_RFLIBCornerPoolCustomOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
