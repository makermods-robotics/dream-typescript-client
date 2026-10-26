/** This package's own version. */
export const VERSION = "0.1.0";

/** The upstream Python releases whose behaviour this package follows. */
export const UPSTREAM = {
  /** `dreamlabs-client` on PyPI (module `dream`): the client, its wire protocol and its post-processing. */
  dreamlabsClient: "1.0.8",
  /** `dreamlabs` on PyPI: `dream_ik`'s arm specs, URDFs and chunk solve. */
  dreamlabs: "1.0.17",
} as const;

/**
 * What every request names in `Dream-Client-Version` (`dream.transport.CLIENT_VERSION_HEADER`) and in the `User-Agent`
 * product `dream-client/<version>`: this package and the upstream client whose wire it speaks, deliberately not a
 * `MAJOR.MINOR.PATCH` version. The gateway serves a checkpoint whose robot has parts only to a semver no older than
 * the first library that converts parts (`shared.contracts.envelope.BadRequestReason`), and this package does not
 * convert parts: claiming `1.0.8` could get such a checkpoint served to a client that cannot read it. Any other shape
 * passes the gateway's ordinary major-version check. The MakerMods console names itself `makermods-console-1.0.8` for
 * the same reason.
 */
export const CLIENT_VERSION = `makermods-dreamlabs-ts-${VERSION}+dream-${UPSTREAM.dreamlabsClient}`;
